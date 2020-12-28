#include <fstream>
#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/attr.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "inference/common/IOBuffer.h"
#include "inference/decoder/Decoder.h"
#include "libraries/decoder/Utils.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"

using namespace w2l;
using namespace w2l::streaming;
namespace py = pybind11;

// -- stuff copied from inference/examples/Utils -- 

namespace cereal {

template <typename Archive>
inline std::string save_minimal(
    const Archive&,
    const w2l::CriterionType& criterionType) {
  switch (criterionType) {
    case w2l::CriterionType::ASG:
      return "ASG";
    case w2l::CriterionType::CTC:
      return "CTC";
    case w2l::CriterionType::S2S:
      return "S2S";
  }
  throw std::runtime_error(
      "save_minimal() got invalid CriterionType value=" +
      std::to_string(static_cast<int>(criterionType)));
}

template <typename Archive>
void load_minimal(
    const Archive&,
    w2l::CriterionType& obj,
    const std::string& value) {
  if (value == "ASG") {
    obj = w2l::CriterionType::ASG;
  } else if (value == "CTC") {
    obj = w2l::CriterionType::CTC;
  } else if (value == "S2S") {
    obj = w2l::CriterionType::S2S;
  } else {
    throw std::runtime_error(
        "load_minimal() got invalid CriterionType value=" + value);
  }
}

} // namespace cereal

#ifdef _WIN32
constexpr const char* separator = "\\";
#else
constexpr const char* separator = "/";
#endif

std::string GetFullPath(const std::string& fileName, const std::string& path) {
  if (!fileName.empty() && fileName[0] == separator[0]) {
    return fileName;
  }
  const std::string requiredSeperator =
      (*path.rbegin() == separator[0]) ? "" : separator;

  return path + requiredSeperator + fileName;
}

// -- end --

constexpr const float kMaxUint16 = static_cast<float>(0x8000);
constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.

struct InferenceResult {
  std::vector<WordUnit> words;
  int chunk_start_time;
  int chunk_end_time;
  InferenceResult(const std::vector<WordUnit>& wordUnits, int chunkStartTime, int chunkEndTime):
    words(wordUnits.begin(), wordUnits.end()), chunk_start_time(chunkStartTime), chunk_end_time(chunkEndTime) {}
};

struct InferenceStream {

  std::shared_ptr<Sequential> dnnModule;
  std::shared_ptr<streaming::Decoder> decoder;
  std::shared_ptr<ModuleProcessingState> input;
  std::shared_ptr<ModuleProcessingState> output;
  std::shared_ptr<IOBuffer> inputBuffer;
  std::shared_ptr<IOBuffer> outputBuffer;
  int nTokens;
  int audioSampleCount;
  int pendingSampleCount;
  bool isFinished;

  InferenceStream(
    std::shared_ptr<Sequential> dnnModule,
    std::shared_ptr<streaming::Decoder> decoder,
    std::shared_ptr<ModuleProcessingState> input,
    std::shared_ptr<ModuleProcessingState> output,
    int nTokens
  ) : dnnModule(dnnModule), decoder(decoder), input(input), output(output), nTokens(nTokens),
    audioSampleCount(0), pendingSampleCount(0), isFinished(false),
    inputBuffer(input->buffer(0)), outputBuffer(output->buffer(0)) {}

  virtual void submit_audio(const py::bytes& audio) {
    if (isFinished) {
      throw std::runtime_error("Audio submitted after audio ended");
    }
    char* bytesBuffer;
    ssize_t bytesLen;
    PYBIND11_BYTES_AS_STRING_AND_SIZE(audio.ptr(), &bytesBuffer, &bytesLen);
    if (bytesLen % sizeof(int16_t)) {
      throw std::runtime_error("Odd number of audio bytes submitted");
    }
    if (!bytesLen) return;
    auto srcPtr = reinterpret_cast<const int16_t*>(bytesBuffer);
    const int sampleCount = bytesLen / sizeof(int16_t);
    inputBuffer->ensure<float>(sampleCount);
    float* bufferPtr = inputBuffer->tail<float>();
    std::transform(srcPtr, srcPtr + sampleCount, bufferPtr, [](int16_t i) -> float {
        return static_cast<float>(i) / kMaxUint16;
      });
    inputBuffer->move<float>(sampleCount);
    pendingSampleCount += sampleCount;
    dnnModule->run(input);
  }

  virtual void end_audio() {
    if (isFinished) return;
    isFinished = true;
    dnnModule->finish(input);
  }

  virtual std::unique_ptr<InferenceResult> next_result(int lookBack = 0) {
    float* data = outputBuffer->data<float>();
    int size = outputBuffer->size<float>();
    if (data && size > 0) {
      decoder->run(data, size);
    }
    if (isFinished) {
      decoder->finish();
    }
    auto words = decoder->getBestHypothesisInWords(lookBack);
    const int chunk_start_ms =
        (audioSampleCount / (kAudioWavSamplingFrequency / 1000));
    const int chunk_end_ms =
        ((audioSampleCount + pendingSampleCount) /
        (kAudioWavSamplingFrequency / 1000));
    audioSampleCount += pendingSampleCount;
    pendingSampleCount = 0;
    // Consume and prune
    const int nFramesOut = outputBuffer->size<float>() / nTokens;
    outputBuffer->consume<float>(nFramesOut * nTokens);
    return std::unique_ptr<InferenceResult>(new InferenceResult(words, chunk_start_ms, chunk_end_ms));
  }

  virtual void prune(int lookBack = 0) {
    decoder->prune(lookBack);
  }

};

struct Model {

  std::shared_ptr<Sequential> dnnModule;
  std::shared_ptr<streaming::Decoder> decoder;
  int nTokens;

  Model(
      std::shared_ptr<Sequential> dnnModule,
      std::shared_ptr<streaming::Decoder> decoder,
      int nTokens)
      : dnnModule(dnnModule), decoder(decoder), nTokens(nTokens) {}

  virtual std::unique_ptr<InferenceStream> open_stream() {
    auto input = std::make_shared<ModuleProcessingState>(1);
    auto output = dnnModule->start(input);
    decoder->start();
    return std::unique_ptr<InferenceStream>(
        new InferenceStream(dnnModule, decoder, input, output, nTokens));
  }
};

std::unique_ptr<Model> load_model(
    const std::string& input_files_base_path,
    const std::string& feature_module_file,
    const std::string& acoustic_module_file,
    const std::string& tokens_file,
    const std::string& decoder_options_file,
    const std::string& lexicon_file,
    const std::string& language_model_file,
    const std::string& transitions_file,
    const std::string& silence_token
) {
  std::shared_ptr<streaming::Sequential> featureModule;
  std::shared_ptr<streaming::Sequential> acousticModule;

  // Read files
  {
    std::ifstream featFile(
        GetFullPath(feature_module_file, input_files_base_path), std::ios::binary);
    if (!featFile.is_open()) {
      throw std::runtime_error(
          "failed to open feature file=" +
          GetFullPath(feature_module_file, input_files_base_path) + " for reading");
    }
    cereal::BinaryInputArchive ar(featFile);
    ar(featureModule);
  }

  {
    std::ifstream amFile(
        GetFullPath(acoustic_module_file, input_files_base_path), std::ios::binary);
    if (!amFile.is_open()) {
      throw std::runtime_error(
          "failed to open acoustic model file=" +
          GetFullPath(acoustic_module_file, input_files_base_path) + " for reading");
    }
    cereal::BinaryInputArchive ar(amFile);
    ar(acousticModule);
  }

  // String both modeles togthers to a single DNN.
  auto dnnModule = std::make_shared<streaming::Sequential>();
  dnnModule->add(featureModule);
  dnnModule->add(acousticModule);

  std::vector<std::string> tokens;
  {
    std::ifstream tknFile(GetFullPath(tokens_file, input_files_base_path));
    if (!tknFile.is_open()) {
      throw std::runtime_error(
          "failed to open tokens file=" +
          GetFullPath(tokens_file, input_files_base_path) + " for reading");
    }
    std::string line;
    while (std::getline(tknFile, line)) {
      tokens.push_back(line);
    }
  }
  int nTokens = tokens.size();

  auto decoderOptions = std::make_shared<DecoderOptions>();
  {
    std::ifstream decoderOptionsFile(
        GetFullPath(decoder_options_file, input_files_base_path));
    if (!decoderOptionsFile.is_open()) {
      throw std::runtime_error(
          "failed to open decoder options file=" +
          GetFullPath(decoder_options_file, input_files_base_path) + " for reading");
    }
    cereal::JSONInputArchive ar(decoderOptionsFile);
    ar(cereal::make_nvp("beamSize", decoderOptions->beamSize),
       cereal::make_nvp("beamSizeToken", decoderOptions->beamSizeToken),
       cereal::make_nvp("beamThreshold", decoderOptions->beamThreshold),
       cereal::make_nvp("lmWeight", decoderOptions->lmWeight),
       cereal::make_nvp("wordScore", decoderOptions->wordScore),
       cereal::make_nvp("unkScore", decoderOptions->unkScore),
       cereal::make_nvp("silScore", decoderOptions->silScore),
       cereal::make_nvp("eosScore", decoderOptions->eosScore),
       cereal::make_nvp("logAdd", decoderOptions->logAdd),
       cereal::make_nvp("criterionType", decoderOptions->criterionType));
  }

  std::vector<float> transitions;
  if (!transitions_file.empty()) {
    std::ifstream transitionsFile(
        GetFullPath(transitions_file, input_files_base_path), std::ios::binary);
    if (!transitionsFile.is_open()) {
      throw std::runtime_error(
          "failed to open transition parameter file=" +
          GetFullPath(transitions_file, input_files_base_path) + " for reading");
    }
    cereal::BinaryInputArchive ar(transitionsFile);
    ar(transitions);
  }

  std::shared_ptr<const DecoderFactory> decoderFactory;
  {
    decoderFactory = std::make_shared<DecoderFactory>(
        GetFullPath(tokens_file, input_files_base_path),
        GetFullPath(lexicon_file, input_files_base_path),
        GetFullPath(language_model_file, input_files_base_path),
        transitions,
        SmearingMode::MAX,
        silence_token,
        0);
  }

  std::shared_ptr<streaming::Decoder> decoder;
  {
    decoder = std::make_shared<streaming::Decoder>(
        decoderFactory->createDecoder(*decoderOptions));
  }

  return std::unique_ptr<Model>(new Model(dnnModule, decoder, nTokens));
}

PYBIND11_MODULE(_inference, m) {
  m.doc() = R"pbdoc(
        wav2letter streaming inference for Python
        -----------------------------------------

        .. currentmodule:: inference

        .. autosummary::
           :toctree: _generate

           load_model
    )pbdoc";

  py::class_<WordUnit>(m, "WordUnit", py::is_final())
      .def_readwrite("word", &WordUnit::word)
      .def_readwrite("begin_time_frame", &WordUnit::beginTimeFrame)
      .def_readwrite("end_time_frame", &WordUnit::endTimeFrame);

  py::class_<InferenceResult>(m, "InferenceResult", py::is_final())
      .def_readwrite("words", &InferenceResult::words)
      .def_readwrite("chunk_start_time", &InferenceResult::chunk_start_time)
      .def_readwrite("chunk_end_time", &InferenceResult::chunk_end_time);

  py::class_<InferenceStream>(m, "InferenceStream", py::is_final())
      .def(
          "submit_audio",
          &InferenceStream::submit_audio,
          "Submit additional audio bytes (PCM, 16-bit mono 16 kHz without WAV header)",
          py::arg("audio"))
      .def(
          "end_audio",
          &InferenceStream::end_audio,
          "Call when there are no more audio bytes in order to finish")
      .def(
          "next_result",
          &InferenceStream::next_result,
          "Run inference to obtain further ASR results since the last time this was called",
          py::arg("look_back") = 0)
      .def(
          "prune",
          &InferenceStream::prune,
          "Prune the decoder's hypothesis space",
          py::arg("look_back") = 0);

  py::class_<Model>(m, "Model", py::is_final())
      .def(
          "open_stream",
          &Model::open_stream,
          "Stream inference from the specified audio file");

  m.def(
      "load_model",
      &load_model,
      R"pbdoc(
        Load model from the specified files
        )pbdoc",
      py::arg("input_files_base_path") = ".",
      py::arg("feature_module_file") = "feature_extractor.bin",
      py::arg("acoustic_module_file") = "acoustic_model.bin",
      py::arg("tokens_file") = "tokens.txt",
      py::arg("decoder_options_file") = "decoder_options.json",
      py::arg("lexicon_file") = "lexicon.txt",
      py::arg("language_model_file") = "language_model.bin",
      py::arg("transitions_file") = "",
      py::arg("silence_token") = "_");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
