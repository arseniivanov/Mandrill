add_mandrill_executable(SceneViewer)

target_sources(SceneViewer
	PRIVATE
		"SceneViewer.cpp"
)

add_shaders(SceneViewer
	"VertexShader.vert"
	"FragmentShader.frag"
	"FragmentShader_VQ.frag"
  "FragmentShader_Raw.frag"
)
