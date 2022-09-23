#include "basic-dnn-cpp/dnn.h"

#include <vector>

double sigmoid (double d) {
    return 1 / exp(d + 1);
}

struct InputToken {
    std::vector<unsigned char> data;
};

struct OutputToken {
    std::vector<double> data;
};

class img_dnn : public DNN<InputToken, OutputToken, sigmoid> {
    public:
        img_dnn (int i, int h, int o, int n, int s);

        void process_input_token (InputToken input_token);

        OutputToken extract_output_token ();

        void fill_output_delta (OutputToken correct);
};

img_dnn::img_dnn (int i, int h, int o, int n, int s) 
    : DNN<InputToken, OutputToken, sigmoid> (i, h, o, n, s) {}

void img_dnn::process_input_token (InputToken input_token) {
    int i = 0;
    for (int k = 0; k < int(input_token.data.size()); k += 4) {
        input_layer[i] = 0.299 * double(input_token.data[k]) 
            + 0.587 * double(input_token.data[k + 1]) 
            + 0.114 * double(input_token.data[k + 2]);
        input_layer[i] /= 256;
        i += 1;
    }
}

OutputToken img_dnn::extract_output_token () {
    std::vector<double> vec;
    OutputToken o;
    for (int i = 0; i < o_layer_sz; i++) {
        o.data.push_back(output_layer[i]);
    }

    return o;
}

void img_dnn::fill_output_delta (OutputToken correct) {
    for (int i = 0; i < o_layer_sz; i++) {
        output_layer[i] = correct.data[i];
    }
}
