#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= -1.616862177848816) {
                            if (x[2] <= -1.0099883079528809) {
                                if (x[2] <= -1.458621084690094) {
                                    return 0;
                                }

                                else {
                                    return 1;
                                }
                            }

                            else {
                                if (x[2] <= 1.4719481468200684) {
                                    return 0;
                                }

                                else {
                                    return 1;
                                }
                            }
                        }

                        else {
                            if (x[0] <= 0.8409156501293182) {
                                if (x[1] <= -2.080846905708313) {
                                    return 1;
                                }

                                else {
                                    return 0;
                                }
                            }

                            else {
                                if (x[1] <= 0.9919107556343079) {
                                    return 0;
                                }

                                else {
                                    return 1;
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }
