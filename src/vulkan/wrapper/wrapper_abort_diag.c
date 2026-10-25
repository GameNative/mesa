#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <unwind.h>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>

void wrapper_i32_note(const char *fmt, ...);

static void
emit(const char *fmt, ...)
{
   char line[512];
   va_list ap;
   va_start(ap, fmt);
   int n = vsnprintf(line, sizeof(line) - 1, fmt, ap);
   va_end(ap);
   if (n < 0)
      return;
   if (n > (int)sizeof(line) - 2)
      n = sizeof(line) - 2;
   __android_log_write(ANDROID_LOG_ERROR, "I32BT", line);
   line[n] = '\n';
   write(2, "err:I32BT ", 10);
   write(2, line, n + 1);
}

static struct sigaction prev_abrt, prev_segv;

struct bt_state {
   int depth;
};

static _Unwind_Reason_Code
bt_frame(struct _Unwind_Context *ctx, void *arg)
{
   struct bt_state *st = arg;
   uintptr_t pc = _Unwind_GetIP(ctx);
   Dl_info info;
   if (!pc || st->depth >= 48)
      return _URC_END_OF_STACK;
   if (dladdr((void *)pc, &info) && info.dli_fname) {
      const char *base = strrchr(info.dli_fname, '/');
      emit("#%02d pc %p %s+0x%lx (%s+0x%lx)",
                          st->depth, (void *)pc, base ? base + 1 : info.dli_fname,
                          (unsigned long)(pc - (uintptr_t)info.dli_fbase),
                          info.dli_sname ? info.dli_sname : "?",
                          info.dli_saddr ? (unsigned long)(pc - (uintptr_t)info.dli_saddr) : 0ul);
   } else {
      emit("#%02d pc %p ?", st->depth, (void *)pc);
   }
   st->depth++;
   return _URC_NO_REASON;
}

static void
diag_handler(int sig, siginfo_t *si, void *uc)
{
   struct sigaction *prev = sig == SIGABRT ? &prev_abrt : &prev_segv;
   struct bt_state st = { 0 };
   emit("signal %d tid %d addr %p", sig, gettid(),
                       si ? si->si_addr : NULL);
   _Unwind_Backtrace(bt_frame, &st);
   if (prev->sa_flags & SA_SIGINFO) {
      if (prev->sa_sigaction)
         prev->sa_sigaction(sig, si, uc);
   } else if (prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
      prev->sa_handler(sig);
   } else {
      sigaction(sig, prev, NULL);
      raise(sig);
   }
}

__attribute__((constructor)) static void
wrapper_abort_diag_init(void)
{
   struct sigaction sa;
   memset(&sa, 0, sizeof(sa));
   sa.sa_sigaction = diag_handler;
   sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
   sigemptyset(&sa.sa_mask);
   sigaction(SIGABRT, &sa, &prev_abrt);
   emit("abort backtrace handler installed");
}

void
wrapper_i32_note(const char *fmt, ...)
{
   char line[400];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(line, sizeof(line), fmt, ap);
   va_end(ap);
   emit("%s", line);
}
