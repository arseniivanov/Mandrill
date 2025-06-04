#pragma once

#if defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(_WIN64)
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_vulkan.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#ifdef MANDRILL_USE_OPENVDB
#include <openvdb/openvdb.h>
#endif

#include <algorithm>
#include <array>
#include <concepts>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <source_location>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#define MANDRILL_NAME "Mandrill"

// Version (set in CMakeLists.txt)
#ifndef MANDRILL_VERSION_MAJOR
#error "Missing major version number"
#endif
#ifndef MANDRILL_VERSION_MINOR
#error "Missing minor version number"
#endif
#ifndef MANDRILL_VERSION_PATCH
#error "Missing patch version number"
#endif

// Platform
#define MANDRILL_PLATFORM_WINDOWS 1
#define MANDRILL_PLATFORM_LINUX 2

#ifndef MANDRILL_PLATFORM
#if defined(_WIN64)
#define MANDRILL_PLATFORM MANDRILL_PLATFORM_WINDOWS
#elif defined(__linux__)
#define MANDRILL_PLATFORM MANDRILL_PLATFORM_LINUX
#else
#error "Unsupported target platform"
#endif
#endif

#define MANDRILL_WINDOWS (MANDRILL_PLATFORM == MANDRILL_PLATFORM_WINDOWS)
#define MANDRILL_LINUX (MANDRILL_PLATFORM == MANDRILL_PLATFORM_LINUX)

// Library export
#if MANDRILL_WINDOWS
#define MANDRILL_API_EXPORT __declspec(dllexport)
#define MANDRILL_API_IMPORT __declspec(dllimport)
#elif MANDRILL_LINUX
#define MANDRILL_API_EXPORT __attribute__((visibility("default")))
#define MANDRILL_API_IMPORT
#endif

#ifdef MANDRILL_DLL
#define MANDRILL_API MANDRILL_API_EXPORT
#else
#define MANDRILL_API MANDRILL_API_IMPORT
#endif

// Use this to disallow copying and default construction of class
#define MANDRILL_NON_COPYABLE(x)                                                                                       \
    MANDRILL_API x() = delete;                                                                                         \
    MANDRILL_API x(const x&) = delete;                                                                                 \
    MANDRILL_API x& operator=(const x&) = delete;

// Some Mandrill specific helper types and classes
namespace Mandrill
{
    // Shorthand for std::shared_ptr<T>
    template <typename T> using ptr = std::shared_ptr<T>;

    // Shorthand for std::make_shared<T>
    template <typename T, typename... Args> static inline ptr<T> make_ptr(Args&&... args)
    {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    template <typename T>
    concept Sizable = requires(T t) {
        { t.size() } -> std::convertible_to<std::size_t>;
    };

    // Most vector counts are uint32_t in Vulkan (use with any container that provides size())
    template <Sizable T> static inline uint32_t count(const T& t)
    {
        return static_cast<uint32_t>(t.size());
    }

    static inline std::filesystem::path GetExecutablePath()
    {
        std::filesystem::path path;
#if MANDRILL_WINDOWS
        TCHAR szFile[260] = {0};
        GetModuleFileNameA(NULL, szFile, 260);
        path = std::filesystem::path(szFile);
#elif MANDRILL_LINUX
        path = std::filesystem::canonical("/proc/self/exe");
#endif
        return path.remove_filename();
    }

#if MANDRILL_LINUX
#define LPCSTR const char*
#endif
    /// <summary>
    /// Open an "Open File" dialogue.
    /// </summary>
    /// <param name="pWindow">GLFW window pointer</param>
    /// <param name="filter">(Windows only) File types to filter on. Check GetOpenFileNameA() and OPENFILENAME:
    /// lpstrFilter for details.</param>
    /// <returns>Path to opened file, empty string on cancel or failure</returns>
    static inline std::string OpenFile(GLFWwindow* pWindow, LPCSTR filter)
    {
#if MANDRILL_WINDOWS
        OPENFILENAME ofn;
        TCHAR szFile[260] = {0};

        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = glfwGetWin32Window(pWindow);
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = filter;
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = NULL;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = NULL;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        if (GetOpenFileNameA(&ofn)) {
            return std::string(ofn.lpstrFile);
        }

#elif MANDRILL_LINUX
        char filename[1024];
        char current_dir_cstr[1024]; // Or use PATH_MAX from <limits.h> for more robustness
        getcwd(current_dir_cstr, sizeof(current_dir_cstr));
        std::string command = "zenity --file-selection --modal --title=\"Select file\" --filename=\"";
        command += current_dir_cstr;

        if (command.back() != '/') {
            command += "/";
        }
        command += "\"";

        FILE* f = popen(command.c_str(), "r");

        if (f) {
            fgets(filename, sizeof(filename), f);
            filename[std::strcspn(filename, "\n")] = 0; // Remove trailing new line
            pclose(f);
            return std::string(filename);
        }
#endif
        return "";
    }

} // namespace Mandrill
