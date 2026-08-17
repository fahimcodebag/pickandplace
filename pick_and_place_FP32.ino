/*
 * ESP32 TFLite Inference - FLOAT32 VERSION (MEMORY OPTIMIZED)
 * 
 * FIXES:
 *   - STATE_PAYLOAD_SIZE corrected from 194 to 190 bytes
 *   - Tensor arena reduced to 100KB (fits in typical ESP32 memory)
 *   - Better memory diagnostics
 * 
 * Float32 model trade-offs:
 *   - Larger model size (~200 KB)
 *   - Slower inference (~30ms vs ~15ms for INT8)
 *   + Perfect accuracy (no quantization loss)
 *   + Simpler code (no quantization parameters)
 */

#include <ArduTFLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your converted FLOAT32 TFLite model
#include "actor_model_float32.h"

// Model parameters
#define STATE_DIM 46
#define ACTION_DIM 7

// Protocol constants
#define STATE_MSG 0x01
#define ACTION_MSG 0x02
#define SYNC_BYTE_0 0xAA
#define SYNC_BYTE_1 0x55
#define SYNC_BYTE_2 0xAA
#define SYNC_BYTE_3 0x55

// Message sizes (AFTER sync pattern)
// CRITICAL: State = type(1) + seq(1) + len(2) + state(184) + crc(2) = 190
// Action = type(1) + seq(1) + len(2) + action(28) + crc(2) = 34
#define STATE_PAYLOAD_SIZE 190
#define ACTION_PAYLOAD_SIZE 34

// TFLite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena sized for ESP32
  // Float32 models need ~80-120KB typically
  constexpr int kTensorArenaSize = 100 * 1024;  // 100KB
  uint8_t* tensor_arena = nullptr;
  
  size_t actual_arena_used = 0;
}

// Statistics
struct Stats {
  uint32_t total_inferences = 0;
  uint32_t successful_inferences = 0;
  uint32_t protocol_errors = 0;
  uint32_t inference_errors = 0;
  uint32_t sync_errors = 0;
  float avg_inference_time_ms = 0.0f;
} stats;

// Function prototypes
bool waitForSync();
bool setupTFLite();
bool receiveState(float* state, uint8_t* seq_num);
void sendAction(const float* action, uint8_t seq_num);
uint16_t calculateChecksum(const uint8_t* data, size_t len);
void runInference(const float* state, float* action);
void printMemoryInfo();

void setup() {
  Serial.begin(921600);
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("\n===========================================");
  Serial.println("ESP32 TFLite Actor - FLOAT32 (OPTIMIZED)");
  Serial.println("===========================================");
  
  // Initial memory status
  printMemoryInfo();
  
  // Check if we have enough memory
  size_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  Serial.printf("\nTensor arena needed: %d KB\n", kTensorArenaSize / 1024);
  Serial.printf("Largest free block: %d KB\n", largest_block / 1024);
  
  if (largest_block < kTensorArenaSize) {
    Serial.println("\n✗ WARNING: Largest block smaller than arena!");
    Serial.printf("  Need: %d KB, Have: %d KB\n", 
                  kTensorArenaSize / 1024, largest_block / 1024);
    Serial.println("  Trying anyway (may fail)...");
  }
  
  // Try PSRAM first (if available)
  #ifdef BOARD_HAS_PSRAM
    if (psramFound()) {
      Serial.println("\nPSRAM detected, using PSRAM for tensor arena...");
      tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
      if (tensor_arena != nullptr) {
        Serial.println("✓ Allocated from PSRAM");
      }
    }
  #endif
  
  // Fallback to heap if PSRAM not available or allocation failed
  if (tensor_arena == nullptr) {
    Serial.println("\nAllocating from heap (DRAM)...");
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    if (tensor_arena != nullptr) {
      Serial.printf("✓ Allocated %d KB from heap\n", kTensorArenaSize / 1024);
    }
  }
  
  if (tensor_arena == nullptr) {
    Serial.println("\n✗✗✗ FATAL: Failed to allocate tensor arena! ✗✗✗");
    Serial.printf("\nRequested: %d KB\n", kTensorArenaSize / 1024);
    Serial.printf("Free heap: %d KB\n", ESP.getFreeHeap() / 1024);
    Serial.printf("Largest block: %d KB\n", largest_block / 1024);
    
    Serial.println("\n--- SOLUTIONS ---");
    Serial.println("1. Reduce kTensorArenaSize in code (try 80KB or 60KB)");
    Serial.println("2. Use ESP32 with PSRAM");
    Serial.println("3. Use INT8 quantized model (needs only ~50KB)");
    Serial.println("4. Close other programs/restart ESP32");
    
    while (1) {
      delay(1000);
      Serial.print(".");
    }
  }
  
  // Setup TensorFlow Lite
  Serial.println("\nInitializing TensorFlow Lite...");
  if (!setupTFLite()) {
    Serial.println("✗ FATAL: TFLite setup failed!");
    Serial.println("\nCheck:");
    Serial.println("  - actor_model_float32.h file exists");
    Serial.println("  - Model is Float32 (not INT8)");
    Serial.println("  - Model schema version matches");
    while (1) delay(1000);
  }
  
  Serial.println("✓ TFLite initialized successfully");
  
  // Memory usage report
  Serial.println("\n--- MEMORY REPORT ---");
  Serial.printf("Tensor arena allocated: %d KB\n", kTensorArenaSize / 1024);
  Serial.printf("Tensor arena used: %d KB (%.1f%%)\n", 
                actual_arena_used / 1024,
                (float)actual_arena_used / kTensorArenaSize * 100);
  Serial.printf("Remaining in arena: %d KB\n", 
                (kTensorArenaSize - actual_arena_used) / 1024);
  
  // Model info
  Serial.println("\n--- MODEL INFO ---");
  Serial.printf("Input: [1, %d] %s\n", STATE_DIM,
                input->type == kTfLiteFloat32 ? "FLOAT32" : "UNKNOWN");
  Serial.printf("Output: [1, %d] %s\n", ACTION_DIM,
                output->type == kTfLiteFloat32 ? "FLOAT32" : "UNKNOWN");
  
  if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
    Serial.println("\n✗ ERROR: Model is NOT Float32!");
    Serial.println("  This code requires Float32 TFLite model");
    Serial.println("  Use: python convert_float32_tflite.py");
    while (1) delay(1000);
  }
  
  // Protocol info
  Serial.println("\n--- PROTOCOL INFO ---");
  Serial.printf("STATE_PAYLOAD_SIZE: %d bytes\n", STATE_PAYLOAD_SIZE);
  Serial.printf("ACTION_PAYLOAD_SIZE: %d bytes\n", ACTION_PAYLOAD_SIZE);
  Serial.printf("Baudrate: %d\n", 921600);
  
  printMemoryInfo();
  
  Serial.println("\n✓✓✓ READY FOR INFERENCE ✓✓✓");
  Serial.println("===========================================\n");
  
  // Clear any startup junk
  delay(100);
  while (Serial.available()) {
    Serial.read();
  }
}

void loop() {
  static float state[STATE_DIM];
  static float action[ACTION_DIM];
  static uint8_t seq_num = 0;
  
  if (receiveState(state, &seq_num)) {
    unsigned long inference_start = micros();
    
    runInference(state, action);
    
    unsigned long inference_time = micros() - inference_start;
    
    stats.total_inferences++;
    stats.successful_inferences++;
    float inference_ms = inference_time / 1000.0f;
    stats.avg_inference_time_ms = 
      (stats.avg_inference_time_ms * (stats.total_inferences - 1) + inference_ms) 
      / stats.total_inferences;
    
    sendAction(action, seq_num);
    
    // Print stats every 100 inferences
    if (stats.total_inferences % 100 == 0) {
      Serial.printf("[%d] Avg: %.2f ms | Heap: %d KB | Action: [%.3f, %.3f, ...]\n",
                    stats.total_inferences, 
                    stats.avg_inference_time_ms, 
                    ESP.getFreeHeap() / 1024,
                    action[0], action[1]);
    }
  }
}

bool setupTFLite() {
  model = tflite::GetModel(actor_model_float32_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("✗ Model schema %d != expected %d\n", 
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  
  static tflite::AllOpsResolver resolver;
  
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("✗ AllocateTensors() failed");
    Serial.println("  Model may be too large for arena");
    return false;
  }
  
  actual_arena_used = interpreter->arena_used_bytes();
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  if (input->dims->size != 2 || input->dims->data[1] != STATE_DIM) {
    Serial.printf("✗ Wrong input shape: expected [1,%d]\n", STATE_DIM);
    return false;
  }
  
  if (output->dims->size != 2 || output->dims->data[1] != ACTION_DIM) {
    Serial.printf("✗ Wrong output shape: expected [1,%d]\n", ACTION_DIM);
    return false;
  }
  
  return true;
}

void printMemoryInfo() {
  Serial.println("\n--- MEMORY STATUS ---");
  Serial.printf("Free heap: %d KB\n", ESP.getFreeHeap() / 1024);
  Serial.printf("Largest block: %d KB\n", 
                heap_caps_get_largest_free_block(MALLOC_CAP_8BIT) / 1024);
  
  #ifdef BOARD_HAS_PSRAM
    if (psramFound()) {
      Serial.printf("PSRAM free: %d KB\n", ESP.getFreePsram() / 1024);
    }
  #endif
  
  Serial.println("---------------------");
}

bool waitForSync() {
  uint8_t sync_state = 0;
  unsigned long start_time = millis();
  const unsigned long SYNC_TIMEOUT = 5000;
  
  while (millis() - start_time < SYNC_TIMEOUT) {
    if (Serial.available() > 0) {
      uint8_t byte_read = Serial.read();
      
      if (sync_state == 0 && byte_read == SYNC_BYTE_0) {
        sync_state = 1;
      } else if (sync_state == 1 && byte_read == SYNC_BYTE_1) {
        sync_state = 2;
      } else if (sync_state == 2 && byte_read == SYNC_BYTE_2) {
        sync_state = 3;
      } else if (sync_state == 3 && byte_read == SYNC_BYTE_3) {
        return true;
      } else {
        sync_state = (byte_read == SYNC_BYTE_0) ? 1 : 0;
      }
    }
  }
  
  stats.sync_errors++;
  return false;
}

bool receiveState(float* state, uint8_t* seq_num) {
  if (!waitForSync()) {
    stats.protocol_errors++;
    return false;
  }
  
  uint8_t buffer[STATE_PAYLOAD_SIZE];
  
  unsigned long start_time = millis();
  size_t bytes_read = 0;
  
  while (bytes_read < STATE_PAYLOAD_SIZE && (millis() - start_time < 1000)) {
    if (Serial.available() > 0) {
      buffer[bytes_read++] = Serial.read();
    }
  }
  
  if (bytes_read != STATE_PAYLOAD_SIZE) {
    stats.protocol_errors++;
    return false;
  }
  
  // Parse header
  uint8_t msg_type = buffer[0];
  *seq_num = buffer[1];
  
  if (msg_type != STATE_MSG) {
    stats.protocol_errors++;
    return false;
  }
  
  // Verify checksum
  uint16_t received_crc = buffer[STATE_PAYLOAD_SIZE - 2] | 
                          (buffer[STATE_PAYLOAD_SIZE - 1] << 8);
  uint16_t calculated_crc = calculateChecksum(buffer, STATE_PAYLOAD_SIZE - 2);
  
  if (received_crc != calculated_crc) {
    stats.protocol_errors++;
    return false;
  }
  
  // Extract Float32 state
  memcpy(state, &buffer[4], STATE_DIM * sizeof(float));
  
  return true;
}

void sendAction(const float* action, uint8_t seq_num) {
  uint8_t buffer[ACTION_PAYLOAD_SIZE];
  
  buffer[0] = ACTION_MSG;
  buffer[1] = seq_num;
  uint16_t payload_len = ACTION_DIM * sizeof(float);
  buffer[2] = payload_len & 0xFF;
  buffer[3] = (payload_len >> 8) & 0xFF;
  
  memcpy(&buffer[4], action, ACTION_DIM * sizeof(float));
  
  uint16_t crc = calculateChecksum(buffer, ACTION_PAYLOAD_SIZE - 2);
  buffer[ACTION_PAYLOAD_SIZE - 2] = crc & 0xFF;
  buffer[ACTION_PAYLOAD_SIZE - 1] = (crc >> 8) & 0xFF;
  
  Serial.write(SYNC_BYTE_0);
  Serial.write(SYNC_BYTE_1);
  Serial.write(SYNC_BYTE_2);
  Serial.write(SYNC_BYTE_3);
  Serial.write(buffer, ACTION_PAYLOAD_SIZE);
  Serial.flush();
}

uint16_t calculateChecksum(const uint8_t* data, size_t len) {
  uint32_t sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum += data[i];
  }
  return sum % 65536;
}

void runInference(const float* state, float* action) {
  // Copy state to input tensor
  for (int i = 0; i < STATE_DIM; i++) {
    input->data.f[i] = state[i];
  }
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    stats.inference_errors++;
    memset(action, 0, ACTION_DIM * sizeof(float));
    return;
  }
  
  // Copy output and clip
  for (int i = 0; i < ACTION_DIM; i++) {
    action[i] = output->data.f[i];
    action[i] = constrain(action[i], -1.0f, 1.0f);
  }
}
