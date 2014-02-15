/* This file contains the function complex_volume */
/* 11/30/09 Matthias Goerner */

#ifndef _complex_volume_
#define _complex_volume_

#include "kernel.h"

extern Complex complex_volume(Triangulation *,
			      const char** err_msg,
			      int *precision,
			      Complex (*dilog_callback)(Complex z));
#endif
