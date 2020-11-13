
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include "model_loader.hpp"
#include <iostream>
#include <arpa/inet.h>
#include <dirent.h>
#include <stdio.h>
#include<thread>

using namespace tensorflow;
using namespace std::chrono;

struct Result
{
    std::string image_name;
    float score;
    int xmin, ymin, xmax, ymax;
};

const std::string PATH_TO_SAVED_MODEL = "/home/ubuntu/unbeatables/dataset/LARC2020/exported/my_mobilenet_best/saved_model";
const std::string PATH_TO_INFERENCE_FILES = "mAP/input/detection-results/";
const std::string IMAGE_PATH = "/home/ubuntu/unbeatables/dataset/LARC2020/dataset/";

void inference(std::vector<std::string> &image_names, ModelLoader &model)
{
    
    Prediction out_pred;
    out_pred.boxes = unique_ptr<vector<vector<float>>>(new vector<vector<float>>());
    out_pred.scores = unique_ptr<vector<float>>(new vector<float>());
    out_pred.labels = unique_ptr<vector<int>>(new vector<int>());
    unsigned int width, height;
    Result result;

    for (int i = 0; i < image_names.size(); ++i)
    // for (int i = 0; i < 2000; ++i)
    {
        cv::Mat opencv_img;
        opencv_img = cv::imread(image_names[i]);
        int width = opencv_img.cols;
        int height = opencv_img.rows;

        // std::cout << "w: " << width << "h: " << height << std::endl;
        model.predict(image_names[i], out_pred);
        for (auto &score : (*out_pred.scores))
        {
            if (score < 0.45)
            {
                continue;
            }
            size_t pos = &score - &(*out_pred.scores)[0];

            auto box = (*out_pred.boxes)[pos];
            result.ymin = (int)(box[0] * height);
            result.xmin = (int)(box[1] * width);
            result.ymax = (int)(box[2] * height);
            result.xmax = (int)(box[3] * width);

            char delimiter = '/';
            char delimiter2 = '.';
            size_t pos1 = image_names[i].rfind(delimiter)+1;
            size_t pos2 = image_names[i].rfind(delimiter2) - pos1;
            std::string token = image_names[i].substr(pos1, pos2);
            ofstream myfile;
            myfile.open("/home/ubuntu/inference_cpp/" + token + ".txt", ios::app);
            myfile << "robot"
                   << " " << score << " " << result.ymin << " " << result.xmin << " " << result.ymax << " " << result.xmax << "\n";
            myfile.close();
        }
        // fim = steady_clock::now();
        // duration<double> time_span = duration_cast<duration<double>>(fim - ini);
        // cout << " - Finalizado em: " << time_span.count() << "s." << endl;
    }
    
}

int main(int argc, char **argv)
{

    std::vector<std::string> image_path;
    std::vector<Result> results;
    steady_clock::time_point ini, fim;

    std::cout << "Loading model...";
    // std::cout << PATH_TO_SAVED_MODEL << std::endl;
    ModelLoader model(PATH_TO_SAVED_MODEL);
    std::cout << "Model loaded...";

    DIR *d;
    struct dirent *dir;
    d = opendir("/home/ubuntu/unbeatables/dataset/LARC2020/dataset/");
    std::string str;
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            // std::cout << " : " << (dir->d_name) << std::endl;
            str = dir->d_name;
            str = IMAGE_PATH + str;
            // std::cout<<str<<std::endl;
            image_path.push_back(str);
        }
        closedir(d);
    }

    image_path.erase(image_path.begin());
    image_path.erase(image_path.begin());

    std::size_t const half_size = image_path.size() / 2;
    std::vector<std::string> split_lo(image_path.begin(), image_path.begin() + half_size);
    std::vector<std::string> split_hi(image_path.begin() + half_size, image_path.end());

    ini = steady_clock::now();
    std::cout << "Tempo de Inicio:" << ini.time_since_epoch().count() << std::endl;

    std::thread thread_lo(inference, std::ref(split_lo), std::ref(model));
    std::thread thread_hi(inference, std::ref(split_hi), std::ref(model)); 
    thread_lo.join();
    thread_hi.join();

    fim = steady_clock::now();

    std::cout << "Finalizado no tempo: " << fim.time_since_epoch().count() << std::endl;
    duration<double> time_span = duration_cast<duration<double>>(fim - ini);
    cout << " - Finalizado em: " << time_span.count() << "s." << endl;


    // inference(image_path, model);
}

// tempo de inferencia: ~33 ms por img