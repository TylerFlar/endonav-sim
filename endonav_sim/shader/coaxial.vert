#version 330

uniform mat4 mvp;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_color;

void main() {
    v_world_pos = in_position;
    v_normal = in_normal;
    v_color = in_color;
    gl_Position = mvp * vec4(in_position, 1.0);
}
