add_mandrill_executable(SDF)

target_sources(SDF
	PRIVATE
		"SDF.cpp"
)

add_shaders(SDF
	"Fullscreen.vert"
	"SDF.frag"
)
