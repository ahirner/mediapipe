// Copyright 2020 Lisandro Bravo.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

class OpenCvVideoImShowCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
 // Unlike MediaPipe convention, Process only runs in MainThread, at least on Mac OS
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status SetupVideoShow();

};

::mediapipe::Status OpenCvVideoImShowCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("VIDEO"));
  cc->Inputs().Tag("VIDEO").Set<ImageFrame>();
  if (cc->Inputs().HasTag("VIDEO_PRESTREAM")) {
    cc->Inputs().Tag("VIDEO_PRESTREAM").Set<VideoHeader>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status OpenCvVideoImShowCalculator::Open(CalculatorContext* cc) {
  
  return SetupVideoShow();
}

::mediapipe::Status OpenCvVideoImShowCalculator::Process(
    CalculatorContext* cc) {
  if (cc->InputTimestamp() == Timestamp::PreStream()) {
    return SetupVideoShow();
  }

  const ImageFrame& image_frame =
      cc->Inputs().Tag("VIDEO").Value().Get<ImageFrame>();
  ImageFormat::Format format = image_frame.Format();
  cv::Mat frame;
  if (format == ImageFormat::GRAY8) {
    frame = formats::MatView(&image_frame);
    if (frame.empty()) {
      return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Receive empty frame at timestamp "
             << cc->Inputs().Tag("VIDEO").Value().Timestamp()
             << " in OpenCvVideoImShowCalculator::Process()";
    }
  } else {
    cv::Mat tmp_frame = formats::MatView(&image_frame);
    if (tmp_frame.empty()) {
      return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Receive empty frame at timestamp "
             << cc->Inputs().Tag("VIDEO").Value().Timestamp()
             << " in OpenCvVideoImShowCalculator::Process()";
    }
    if (format == ImageFormat::SRGB) {
      cv::cvtColor(tmp_frame, frame, cv::COLOR_RGB2BGR);
    } else if (format == ImageFormat::SRGBA) {
      cv::cvtColor(tmp_frame, frame, cv::COLOR_RGBA2BGR);
    } else {
      return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Unsupported image format: " << format;
    }
  }
  cv::imshow("MediaPipe",frame);
  cv::waitKey(1);
  
  return ::mediapipe::OkStatus();
}

::mediapipe::Status OpenCvVideoImShowCalculator::Close(CalculatorContext* cc) {
  cv::destroyWindow("MediaPipe");
  return ::mediapipe::OkStatus();
}

::mediapipe::Status OpenCvVideoImShowCalculator::SetupVideoShow() {
  cv::namedWindow("MediaPipe",cv::WINDOW_GUI_EXPANDED);
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(OpenCvVideoImShowCalculator);
}  // namespace mediapipe
