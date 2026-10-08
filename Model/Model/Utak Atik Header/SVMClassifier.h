#pragma once

namespace Eloquent {
    namespace ML {
        namespace Port {
            class SVM {
            public:
                /**
                 * Predict class for features vector
                 */
                int predict(float *x);

            protected:
                /**
                 * Compute kernel between feature vector and support vector.
                 * Kernel type: linear
                 */
                float compute_kernel(float *x, ...);
            };
        }
    }
}
