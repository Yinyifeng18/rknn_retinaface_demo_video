// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
 #include <unistd.h>
#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include "rga.h"
#include "rknn_api.h"
#include <dirent.h>

#include "retinaface.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"



int main(int argc, char **argv)
{

    const char *model_path = "./model/RetinaFace.rknn";

    int ret;
    bool success;  

     // 视频流
    cv::VideoCapture cap("./model/test.mp4");
    if (!cap.isOpened())
    {
        std::cout << "无法打开test.mp4文件" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    // using opencv
    using namespace cv;
    using namespace std;
    cv::Mat img;
    cv::Mat orig_img;
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Video", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // rknn 初始化
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_retinaface_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    while(1)
    {
        /* 读取数据  */
        success = cap.read(orig_img);
        if (!success)  
        {
            break;  
        }
        cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    
        /* 重新组包 */
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = img.cols;
        src_image.height = img.rows;
        src_image.width_stride = img.step[0];
        src_image.height_stride = img.step[1];
        src_image.format = IMAGE_FORMAT_RGB888;      
        src_image.virt_addr = img.data;
        src_image.size = img.total() * img.elemSize();
        src_image.fd = 0;

        printf("src_image.width = %d\n", src_image.width);
	    printf("src_image.height = %d\n", src_image.height);
    	printf("src_image.width_stride = %d\n", src_image.width_stride);
	    printf("src_image.height_stride = %d\n", src_image.height_stride);
    	printf("src_image.format = %d\n", src_image.format);
    	printf("src_image.size = %d\n", src_image.size);
        printf("src_image.fd = %d\n", src_image.fd);


        retinaface_result result;
        ret = inference_retinaface_model(&rknn_app_ctx, &src_image, &result);
        if (ret != 0) {
            printf("init_retinaface_model fail! ret=%d\n", ret);
            goto out;
        }

        for (int i = 0; i < result.count; ++i) {
            int rx = result.object[i].box.left;
            int ry = result.object[i].box.top;
            int rw = result.object[i].box.right - result.object[i].box.left;
            int rh = result.object[i].box.bottom - result.object[i].box.top;
            draw_rectangle(&src_image, rx, ry, rw, rh, COLOR_GREEN, 3);
            char score_text[20];
            snprintf(score_text, 20, "%0.2f", result.object[i].score);
            printf("face @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
            draw_text(&src_image, score_text, rx, ry, COLOR_RED, 20);
            for(int j = 0; j < 5; j++) {
                draw_circle(&src_image, result.object[i].ponit[j].x, result.object[i].ponit[j].y, 2, COLOR_ORANGE, 4);
            }
        }
        //write_image("result.jpg", &src_image);
        
        /* 把数据转成cv格式 */
        orig_img = cv::Mat(src_image.height, src_image.width, CV_8UC3, src_image.virt_addr);
		cv::rotate(orig_img, orig_img, ROTATE_90_COUNTERCLOCKWISE);
        cv::cvtColor(orig_img, img, cv::COLOR_RGB2BGR);
        cv::imshow("Video", img);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
        usleep(20*1000);
    }

out:
    ret = release_retinaface_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_retinaface_model fail! ret=%d\n", ret);
    }



    return 0;
}
