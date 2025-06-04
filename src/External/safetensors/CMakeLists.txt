# Mandrill/src/External/safetensors/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MandrillSafeTensorsLoader)

add_library(mandrill_safetensors_loader # Define a new library target
  SHARED
    model_loader.cc
)

# Make "model_loader.h" (and thus "safetensors.hh") findable
target_include_directories(mandrill_safetensors_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} # This directory
)

# Ensure C++ standard for <regex> in model_loader.cc
set_property(TARGET mandrill_safetensors_loader PROPERTY CXX_STANDARD 17) # Or your preferred modern standard
set_property(TARGET mandrill_safetensors_loader PROPERTY CXX_STANDARD_REQUIRED ON)
