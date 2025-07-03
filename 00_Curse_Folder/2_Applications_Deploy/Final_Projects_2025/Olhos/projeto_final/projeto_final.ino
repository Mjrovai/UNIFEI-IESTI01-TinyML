#include <TinyMLShield.h>
#include <TensorFlowLite.h>

#include "lstm_model.h"
#include "cnn_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


constexpr int kTensorArenaSize = 50 * 1024;
alignas(16) uint8_t tensor_arena_cnn[kTensorArenaSize];
alignas(16) uint8_t tensor_arena_lstm[kTensorArenaSize];

// === CNN ===
const tflite::Model* cnn_model = nullptr;
tflite::MicroInterpreter* cnn_interpreter = nullptr;
TfLiteTensor* cnn_input = nullptr;
TfLiteTensor* cnn_output = nullptr;

// === LSTM ===
const tflite::Model* lstm_model = nullptr;
tflite::MicroInterpreter* lstm_interpreter = nullptr;
TfLiteTensor* lstm_input = nullptr;
TfLiteTensor* lstm_output = nullptr;

// === Configuração da CNN ===
constexpr int FRAME_WIDTH = 96;
constexpr int FRAME_HEIGHT = 96;
constexpr int FRAME_CHANNELS = 1;
constexpr int FEATURE_SIZE = 16;  // <- ex: Flatten após última conv

float features[7][FEATURE_SIZE]; // buffer para sequência de 7 vetores

// === Função para carregar modelo CNN ===
void initCNN() {
  cnn_model = tflite::GetModel(cnn_model_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_cnn_interpreter(
      cnn_model, resolver, tensor_arena_cnn, kTensorArenaSize);
  cnn_interpreter = &static_cnn_interpreter;

  if (cnn_interpreter->AllocateTensors()) {
    Serial.println("Erro ao alocar tensores da CNN!");
    TfLiteStatus status = cnn_interpreter->AllocateTensors();
    while (1);
  }
  TfLiteStatus status = cnn_interpreter->AllocateTensors();

  cnn_input = cnn_interpreter->input(0);
  cnn_output = cnn_interpreter->output(0);
  Serial.println("Modelo CNN deu certo!");
}

// === Função para carregar modelo LSTM ===
void initLSTM() {
  lstm_model = tflite::GetModel(lstm_model_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_lstm_interpreter(
      lstm_model, resolver, tensor_arena_lstm, kTensorArenaSize);
  lstm_interpreter = &static_lstm_interpreter;

  if (lstm_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Erro ao alocar tensores LSTM");
    while (1);
  }

  lstm_input = lstm_interpreter->input(0);
  lstm_output = lstm_interpreter->output(0);
  Serial.println("Modelo LSTM deu certo!");
}

// OV767X Camera = OV767X();
byte image[160 * 120];

// OV7675 Camera;

void initCamera() {
  // Inicia câmera em grayscale com tamanho menor
  if (!Camera.begin(QQVGA, GRAYSCALE, 5, OV7675)) {
    Serial.println("Erro ao iniciar câmera");
    while (1);
  }

  Serial.println("Cameraerâ deu certo!");
}

uint8_t resized_image[96 * 96];

void resizeImageNN(const uint8_t* src, int srcWidth, int srcHeight,
                   uint8_t* dst, int dstWidth, int dstHeight) {
  for (int y = 0; y < dstHeight; y++) {
    int srcY = y * srcHeight / dstHeight;
    for (int x = 0; x < dstWidth; x++) {
      int srcX = x * srcWidth / dstWidth;
      dst[y * dstWidth + x] = src[srcY * srcWidth + srcX];
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Inicializando modelos e dispositivos...");
  
  initCamera();

  initCNN();
  initLSTM();
  
  pinMode(13, OUTPUT);   // Configura o pino 13 como saída (LED integrado)
}

int feature_idx = 0;

void loop() {
  Camera.readFrame(image);
  resizeImageNN(image, 160, 120, resized_image, 96, 96);

  // Normaliza imagem
  for (int i = 0; i < FRAME_WIDTH * FRAME_HEIGHT; i++) {
    //cnn_input->data.int[i] = resized_image[i] / 255;
    cnn_input->data.int8[i] = ((int)resized_image[i]) - 128;
  }

  // Inference CNN
  if (cnn_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Erro na inferência da CNN");
    return;
  } 

  // Salva resultado da CNN no buffer de features
  for (int i = 0; i < FEATURE_SIZE; i++) {
    features[feature_idx][i] = cnn_output->data.int8[i];
    // Serial.print(cnn_output->data.int8[i]);
  }

  Serial.println("CNN executou!");

  feature_idx++;

  // Quando tiver 7 frames, executa LSTM
  if (feature_idx == 7) {
    feature_idx = 0;

    // Preenche input do LSTM com os 7 vetores
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < FEATURE_SIZE; j++) {
        lstm_input->data.int8[i * FEATURE_SIZE + j] = features[i][j];
      }
    }

    if (lstm_interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Erro na inferência do LSTM");
      return;
    }

    int8_t* lstm_result = lstm_output->data.int8;
    int output_size = lstm_output->bytes / sizeof(int8_t);
    Serial.print("Saída LSTM: ");
    for (int i = 0; i < output_size; i++) {
      Serial.print(lstm_result[i], 4);
      Serial.print(" ");
    }
    if (lstm_result[0] < 50) {
      digitalWrite(13, HIGH);
    } else {
      digitalWrite(13, LOW);
    }
    Serial.println();
  }

  delay(200);  // ~4 FPS
}

