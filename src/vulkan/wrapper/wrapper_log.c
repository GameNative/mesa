#include <time.h>
#include <pthread.h>
#include <stdbool.h>

#include "wrapper_log.h"
#include "wrapper_util.h"

void wrapper_diag_append(const char *fmt, ...);

#define PATH_MAX_SIZE 1024
#define WRAPPER_DIR "/sdcard/wrapper"
#define WRAPPER_SHADER_LOG_PATH WRAPPER_DIR "/shaders"
#define WRAPPER_LOG_PATH WRAPPER_DIR "/logs"
#define WRAPPER_VALIDATION_LOG_PATH WRAPPER_DIR "/validation"

static struct wrapper_log wrapper_log_options[] = {
	{"info", WRAPPER_LOG_INFO},
	{"error", WRAPPER_LOG_ERROR},
	{"shader", WRAPPER_LOG_SHADER},
	{"validation", WRAPPER_LOG_VALIDATION},
	{"bcn", WRAPPER_LOG_BCN},
	{"apidump", WRAPPER_LOG_APIDUMP},
	{NULL, 0}
};

uint64_t wrapper_log_mask;

static void get_formatted_date_time(char *buf, size_t length) 
{  
   time_t rawtime = time(NULL);
   struct tm *ptm = localtime(&rawtime);

   strftime(buf, 256, "%F_%H-%M-%S", ptm);
}

char *get_executable_name() {
   char *path = malloc(PATH_MAX);

   int fd = open("/proc/self/cmdline", O_RDONLY);

   if (fd != -1) {
      read(fd, path, PATH_MAX_SIZE);
      char *ptr = strrchr(path, '/');
      if (ptr)
         path = ptr + 1;
      ptr = strrchr(path, '\\');
      if (ptr)
         path = ptr + 1;
      close(fd);
   }
   
   return path;
}

static unsigned long long get_debug_flag(const char *option) {
   int index = 0;

   while (wrapper_log_options[index].name != NULL) {
      if (!strcmp(wrapper_log_options[index].name, option))
         return wrapper_log_options[index].value;
         
      index++;
   }

   return 0;
}

static void parse_wrapper_debug_str(char *wrapper_log_level_env) {
   if (!wrapper_log_level_env) {
      wrapper_log_mask = 0;
      return;
   }

   char *option = strtok(wrapper_log_level_env, ",");

   while (option != NULL) {
      wrapper_log_mask |= get_debug_flag(option);
      option = strtok(NULL, ",");
   }
}

int get_wrapper_log_level(const char *option) {
    uint64_t flag = get_debug_flag(option);

    if (wrapper_log_mask & flag)
       return 1;

    return 0;
}

void write_to_logfile(const char *fmt, const char *level, ...)  {
   static FILE *wrapper_log_file = NULL;
   va_list va_args;

   va_start(va_args, level);

   if (!wrapper_log_file) {
      char *wrapper_log_filename = getenv("WRAPPER_LOG_FILE");
      char date[256];

      get_formatted_date_time(date, 256);
      
      if (!wrapper_log_filename)
         asprintf(&wrapper_log_filename, "%s/%s_%s", WRAPPER_LOG_PATH, get_executable_name(), date);
         
      if (!strcmp("stdout", wrapper_log_filename)) {
         wrapper_log_file = stdout;
       }
       else {
         wrapper_log_file = fopen(wrapper_log_filename, "w");
       } 
   }

   if (wrapper_log_file) {
      fprintf(wrapper_log_file, "[%s]: ", level);
      vfprintf(wrapper_log_file, fmt, va_args);
      fprintf(wrapper_log_file, "\n");
      fflush(wrapper_log_file);
   }

   /* When WRAPPER_DIAG is on, also emit to stderr — which for a D3D process is
    * redirected into the single per-game diag file, so the wrapper's own
    * decisions (faked extensions, disabled features) join the device report and
    * the vkd3d/DXVK output in one shareable file. */
   static int diag = -1;
   if (diag == -1)
      diag = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
   if (diag) {
      va_list va2;
      va_start(va2, level);
      fprintf(stderr, "[%s]: ", level);
      vfprintf(stderr, fmt, va2);
      fprintf(stderr, "\n");
      fflush(stderr);
      va_end(va2);
   }

   va_end(va_args);
}

void init_wrapper_logging()
{
   /* Diagnostic package: keep the existing capability/format/copy probes on
    * even when the launcher does not pass the guest diagnostic environment
    * through to the native OpenGL/Zink driver process. */
   setenv("WRAPPER_DIAG", "1", 1);
   fprintf(stderr, "[WRAPPER_DIAG] isaac-wrapper-validation-3 loaded; validation status and bounded messages captured automatically\n");
   fflush(stderr);

   char *wrapper_log_level_env = getenv("WRAPPER_LOG_LEVEL");
   parse_wrapper_debug_str(wrapper_log_level_env);
   wrapper_log_mask |= WRAPPER_LOG_VALIDATION;

   CREATE_FOLDER(WRAPPER_DIR, 0770);
   CREATE_FOLDER(WRAPPER_LOG_PATH, 0770);
   CREATE_FOLDER(WRAPPER_SHADER_LOG_PATH, 0770);
   CREATE_FOLDER(WRAPPER_VALIDATION_LOG_PATH, 0770);
}

void dump_shader_code(const uint32_t *code, size_t size) {
   char *file; 	
   static int index = 0;
   
   asprintf(&file, "%s/%s_shader_%d.spv", WRAPPER_SHADER_LOG_PATH, get_executable_name(), index); 

   FILE *fp = fopen(file, "wb"); 
   if (fp) {
      fwrite(code, 1, size, fp); 
      fclose(fp); 
   }

   index++;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
wrapper_debug_utils_messenger(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                              const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
                              void *userData)
{
   /* The app collects the shared diagnostic stream, not /sdcard/wrapper/validation.
    * Bound and deduplicate warning/error messages so a per-draw error cannot
    * drown out the report or stall the game. Keep the VUID and operation text. */
   static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
   static int32_t seen_ids[64];
   static unsigned seen_count;
   if (callbackData &&
       (messageSeverity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))) {
      pthread_mutex_lock(&log_mutex);
      bool seen = false;
      for (unsigned i = 0; i < seen_count; i++)
         seen |= seen_ids[i] == callbackData->messageIdNumber;
      if (!seen && seen_count < 64) {
         seen_ids[seen_count++] = callbackData->messageIdNumber;
         wrapper_diag_append("[WRAPPER_VALIDATION] severity=0x%x type=0x%x id=%.256s code=%d: %.8192s\n",
                             messageSeverity, messageTypes,
                             callbackData->pMessageIdName ? callbackData->pMessageIdName : "unknown",
                             callbackData->messageIdNumber,
                             callbackData->pMessage ? callbackData->pMessage : "no text");
         if (seen_count == 64)
            wrapper_diag_append("[WRAPPER_VALIDATION] unique-message limit reached; further messages suppressed\n");
      }
      pthread_mutex_unlock(&log_mutex);
   }
   return VK_FALSE;
}
