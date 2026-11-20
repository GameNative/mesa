#include <stdlib.h>
#include <dlfcn.h>
#include <string>
#include <iostream>
#include <unistd.h>

#include "graphicsenv_hook.hpp"
#include "wrapper_log.h"

#define LIBNAME "/system/lib64/libgraphicsenv.so"
#define WRAPPER_LAYERS_PATH "/data/data/com.winlator.cmod/files/imagefs/usr/lib:/data/data/com.termux/files/usr/lib"

static void *graphicsenv_handle = nullptr;
static std::string env_layers_path;

static void *(*getInstance)() = nullptr;
static const std::string& (*getLayerPaths)(void *) = nullptr;
static void *(*getAppNamespace)(void *) = nullptr;
static void (*setLayerPaths)(void *, void *, const std::string) = nullptr;

template <typename T>
void get_function_pointer(T& pointer, const std::string& name) {
   if (!pointer) {
      pointer = reinterpret_cast<T>(dlsym(graphicsenv_handle, name.c_str()));
   }
}

extern "C"
bool set_layer_paths() {
   if (!graphicsenv_handle)
      graphicsenv_handle = dlopen(LIBNAME, RTLD_NOW);

   get_function_pointer(getInstance, "_ZN7android11GraphicsEnv11getInstanceEv");
   get_function_pointer(getLayerPaths, "_ZN7android11GraphicsEnv13getLayerPathsEv");
   get_function_pointer(getAppNamespace, "_ZN7android11GraphicsEnv15getAppNamespaceEv");
   get_function_pointer(setLayerPaths, "_ZN7android11GraphicsEnv13setLayerPathsEPNS_21NativeLoaderNamespaceERKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE");
   if (!setLayerPaths) {
      get_function_pointer(setLayerPaths, "_ZN7android11GraphicsEnv13setLayerPathsEPNS_21NativeLoaderNamespaceENSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE");
   }

   if (!getInstance || !getLayerPaths || !getAppNamespace || !setLayerPaths) {
      WRAPPER_LOG(error, "Failed to get function pointers from libgraphicsenv.so");
      return false;    	
   }

   void *instance = getInstance();

   if (!instance) {
      WRAPPER_LOG(error, "Failed to obtain GraphicsEnv instance");
      return false;
   }

   void *app_namespace = getAppNamespace(instance);

   if (env_layers_path.empty()) {
      const char *env = getenv("WRAPPER_LAYER_PATH");
      if (env)
         env_layers_path = env;
      if (env_layers_path.empty()) {
         /* Keep app-provided paths and search beside this installed driver;
          * the legacy fallback alone points only at unrelated applications. */
         env_layers_path = getLayerPaths(instance);
         Dl_info module = {};
         if (dladdr(reinterpret_cast<void *>(&set_layer_paths), &module) && module.dli_fname) {
            std::string path(module.dli_fname);
            auto slash = path.find_last_of('/');
            if (slash != std::string::npos) {
               if (!env_layers_path.empty())
                  env_layers_path += ':';
               env_layers_path += path.substr(0, slash);
            }
         }
         if (env_layers_path.empty())
            env_layers_path = WRAPPER_LAYERS_PATH;
      }
   }

   setLayerPaths(instance, app_namespace, env_layers_path);

   auto path = getLayerPaths(instance);

   if (path != env_layers_path) {
      WRAPPER_LOG(error, "Failed to change layer paths");
      return false;
   }

   return true;
}
