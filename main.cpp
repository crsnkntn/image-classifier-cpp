#include <iostream>
#include <filesystem>
#include <vector>

#include "png-lib/lodepng.h"
#include "png-lib/lodepng.cpp"

#include "img-dnn.cpp"

int main () {
    img_dnn net(100*100, 100, 10, 10, SEED);

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    InputToken i_token;
    OutputToken o_token, last;

    for (const auto& dirEntry : recursive_directory_iterator("num-images")) {
        i_token.data.resize(0);
        std::string str = dirEntry.path();
        unsigned width, height;
        const char* image_file = str.c_str();
        std::cout << image_file << std::endl;
        
        unsigned error = lodepng::decode(i_token.data, width, height, image_file);


        if (error) 
            std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

        net.process_input_token(i_token);
        o_token.data.resize(0);
        for (int i = 0; i < 10; i++) {
            if (i == str[9] - 48)
                o_token.data.push_back(1);
            else 
                o_token.data.push_back(0);
        }
        
        net.train_with_current_input(o_token);
        last = net.extract_output_token();
    }
    for (int i = 0; i < 10; i++) {
        std::cout << last.data[i] << std::endl;
    }
}


/*
    void load_png_to_input_layer (const char* image_file) {
        std::vector<unsigned char> image;
        unsigned width, height;

        // Derive the raw bytes, image width & height, check for errors
        unsigned error = lodepng::decode(image, width, height, image_file);
        if (error) 
            std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

        if (int(width * height) < i_layer_sz) {
            std::cout << "image dimension error" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Convert the raw bytes into a grayscale version, stored in the nn input layer
        int i = 0;
        for (int k = 0; k < int(image.size()); k += 4) {
            input_layer[i] = 0.299 * double(image[k]) + 0.587 * double(image[k + 1]) + 0.114 * double(image[k + 2]);
            input_layer[i] /= 256;
            i += 1;
        }
    }*/
