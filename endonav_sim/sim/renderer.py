"""ModernGL offscreen renderer with a custom coaxial-lighting shader.

We bypass pyrender entirely (its shader override is fragile) and drive a
single-pass forward renderer ourselves. The mesh is uploaded once; each
render() updates only the MVP and camera-position uniforms."""

from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
import trimesh

SHADER_DIR = Path(__file__).parent / "shader"


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at_from_pose(pose: np.ndarray) -> np.ndarray:
    """Pose is camera-to-world (4x4). Returns world-to-camera (view matrix)
    using the OpenGL convention: camera looks down -Z, +Y up, +X right."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    # World-to-camera = inverse of pose. For a rigid transform: R^T, -R^T t.
    Rt = R.T
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = Rt
    view[:3, 3] = -Rt @ t
    return view


class CoaxialRenderer:
    # Letterbox layout that matches the real phantom camera output:
    # 870x760 active image area centered (almost) in a 1024x768 frame.
    ACTIVE_W = 870
    ACTIVE_H = 760
    BAR_LEFT = 77  # px of black on the left of the active region
    BAR_TOP = 4  # px of black on the top of the active region

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        width: int = 1024,
        height: int = 768,
        fov_y_deg: float = 95.0,
        near_mm: float = 0.1,
        far_mm: float = 200.0,
        supersample: int = 2,
        chromatic_aberration: float = 0.006,
        sensor_noise: float = 0.06,
    ) -> None:
        # Output size matches the real phantom camera (1024x768, 4:3) with
        # letterbox bars around an 870x760 active image region.
        self.width = width
        self.height = height
        self.fov_y_deg = fov_y_deg
        self.near = near_mm
        self.far = far_mm
        self.chromatic_aberration = chromatic_aberration
        self.sensor_noise = sensor_noise
        # Supersampling AA: render the active region at ss× resolution into
        # an offscreen buffer, then resolve+composite in pass B with chroma
        # aberration, noise, and letterbox bars.
        self.supersample = max(1, int(supersample))
        self.ss_w = self.ACTIVE_W * self.supersample
        self.ss_h = self.ACTIVE_H * self.supersample
        self._frame = 0  # noise seed

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        # No face culling: the mesh has inward-flipped winding for correct
        # vertex normals (lighting), but the resulting screen-space winding
        # under a right-handed view would cull the visible side. Drawing
        # both sides is cheap on a ~100k-tri mesh and avoids the foot-gun.

        vert_src = (SHADER_DIR / "coaxial.vert").read_text()
        frag_src = (SHADER_DIR / "coaxial.frag").read_text()
        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        post_vert = (SHADER_DIR / "postprocess.vert").read_text()
        post_frag = (SHADER_DIR / "postprocess.frag").read_text()
        self.post_prog = self.ctx.program(vertex_shader=post_vert, fragment_shader=post_frag)

        self._upload_mesh(mesh)

        # Pass A: lit scene at the supersampled resolution.
        self.scene_color = self.ctx.texture((self.ss_w, self.ss_h), 4, dtype="f1")
        self.scene_color.repeat_x = False
        self.scene_color.repeat_y = False
        # LINEAR_MIPMAP_LINEAR is overkill; LINEAR is enough for a 2× downsample.
        self.scene_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Depth at the SUPERSAMPLED resolution; we read it back at output res
        # via a separate non-supersampled depth pass would be wasteful — we
        # just sample the center pixel of each output pixel for true depth.
        self.scene_depth = self.ctx.depth_texture((self.ss_w, self.ss_h))
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_color], depth_attachment=self.scene_depth
        )

        # Pass B: output at native resolution (the consumer reads this).
        self.color_tex = self.ctx.texture((width, height), 4, dtype="f1")
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])

        # Fullscreen triangle for the resolve pass.
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.post_prog, [(self.quad_vbo, "2f", "in_position")]
        )

        # FOV applies to the *active* region, not the full letterboxed frame.
        self.proj = _perspective(self.fov_y_deg, self.ACTIVE_W / self.ACTIVE_H, near_mm, far_mm)

        # Lighting defaults — tuned so a wall at ~3 mm reads near saturation
        # and the visible end of a 25 mm tube reads near black.
        self.light_scale = 80.0
        self.spec_strength = 0.4
        self.spec_power = 16.0
        self.spot_exp = 2.5  # EndoPBR spotlight cosine exponent

    def _upload_mesh(self, mesh: trimesh.Trimesh) -> None:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            colors = np.tile(np.array([0.78, 0.47, 0.51], dtype=np.float32), (len(verts), 1))
        faces = np.asarray(mesh.faces, dtype=np.uint32)

        interleaved = np.concatenate([verts, normals, colors], axis=1).astype(np.float32)
        self.vbo = self.ctx.buffer(interleaved.tobytes())
        self.ibo = self.ctx.buffer(faces.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
            index_buffer=self.ibo,
        )

    def render(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Render at the given camera-to-world pose. Returns (rgb_uint8, depth_mm).

        The camera convention is OpenGL: looking down -Z in its local frame.
        For an endoscope advancing along its viewing direction, callers should
        treat the camera's local -Z as 'forward'."""
        view = _look_at_from_pose(pose)
        mvp = (self.proj @ view).astype(np.float32)

        # ---- Pass A: lit scene at supersampled resolution -----------------
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, self.ss_w, self.ss_h)
        self.scene_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        cam_pos = pose[:3, 3].astype(np.float32)
        # Camera forward in world: pose's local -Z column. Normalized.
        cam_fwd = -pose[:3, 2].astype(np.float32)
        cam_fwd = cam_fwd / max(float(np.linalg.norm(cam_fwd)), 1e-8)
        self.prog["mvp"].write(mvp.T.tobytes())  # column-major for GL
        self.prog["cameraPos"].value = tuple(cam_pos.tolist())
        self.prog["cameraForward"].value = tuple(cam_fwd.tolist())
        self.prog["lightScale"].value = float(self.light_scale)
        self.prog["specStrength"].value = float(self.spec_strength)
        self.prog["specPower"].value = float(self.spec_power)
        self.prog["spotExp"].value = float(self.spot_exp)
        self.vao.render(moderngl.TRIANGLES)

        # ---- Pass B: resolve, letterbox, chroma aberration, sensor noise --
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.scene_color.use(location=0)
        self.post_prog["scene_tex"].value = 0
        self.post_prog["supersample"].value = self.supersample
        self.post_prog["texel"].value = (1.0 / self.ss_w, 1.0 / self.ss_h)
        self.post_prog["frame_size"].value = (float(self.width), float(self.height))
        self.post_prog["active_size"].value = (float(self.ACTIVE_W), float(self.ACTIVE_H))
        self.post_prog["bar_offset"].value = (float(self.BAR_LEFT), float(self.BAR_TOP))
        self.post_prog["chroma_strength"].value = float(self.chromatic_aberration)
        self.post_prog["noise_strength"].value = float(self.sensor_noise)
        self.post_prog["frame_id"].value = float(self._frame)
        self._frame += 1
        self.quad_vao.render(moderngl.TRIANGLES)

        rgba = np.frombuffer(self.fbo.read(components=4, dtype="f1"), dtype=np.uint8)
        rgba = rgba.reshape(self.height, self.width, 4)
        rgb = np.flipud(rgba[:, :, :3]).copy()

        # Depth from the supersampled depth buffer (active region only),
        # downsampled by stride and pasted into a letterboxed depth map.
        depth_buf_ss = np.frombuffer(self.scene_depth.read(), dtype=np.float32).reshape(
            self.ss_h, self.ss_w
        )
        s = self.supersample
        depth_active = depth_buf_ss[s // 2 :: s, s // 2 :: s][: self.ACTIVE_H, : self.ACTIVE_W]
        depth_buf = np.ones((self.height, self.width), dtype=np.float32)  # 1.0 = far plane
        depth_buf[
            self.BAR_TOP : self.BAR_TOP + self.ACTIVE_H,
            self.BAR_LEFT : self.BAR_LEFT + self.ACTIVE_W,
        ] = depth_active
        depth_buf = np.flipud(depth_buf).copy()
        # Linearize: GL stores depth as a non-linear [0,1] value in clip space.
        z_ndc = depth_buf * 2.0 - 1.0
        depth_mm = (2.0 * self.near * self.far) / (
            self.far + self.near - z_ndc * (self.far - self.near)
        )
        depth_mm[depth_buf >= 1.0] = np.inf
        return rgb, depth_mm

    def release(self) -> None:
        self.fbo.release()
        self.color_tex.release()
        self.scene_fbo.release()
        self.scene_color.release()
        self.scene_depth.release()
        self.quad_vao.release()
        self.quad_vbo.release()
        self.post_prog.release()
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.prog.release()
        self.ctx.release()
