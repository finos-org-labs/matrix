// Package matrix provides high-performance matrix operations with SIMD optimization.
package matrix

/*
#cgo CFLAGS: -I../../include

#include <matrix.h>
*/
import "C"

import (
	"fmt"
)

// checkStatus converts C status code to Go error
func checkStatus(status C.int, operation string) error {
	if status == 0 {
		return nil
	}
	return fmt.Errorf("%s failed with status %d", operation, int(status))
}
