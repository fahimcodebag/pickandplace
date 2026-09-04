/*
 * ESP32 TFLite Pick-and-Place — INT8, TWO-MODEL FSM
 * ==================================================
 * Runs the decomposed grasp + transport policies (both INT8-quantized) and the
 * rule-based finite state machine that glues them, entirely on the ESP32.
 *
 * The PC keeps running the robosuite simulator (HIL): every control step it
 * sends the 46-float observation over serial; this sketch decides which policy
 * / scripted phase is active, runs the right INT8 model (or a scripted
 * P-controller), and returns the 7-float action. The FSM state lives on the
 * MCU — the PC only steps physics.
 *
 * Ported from Decomposed state training/place_env_wrapper.py (the validated
 * 92% fixed-spawn pipeline). Locked deployment params: trigger 0.14 m /
 * hold 3 / place-horizon 300.
 *
 * ── FSM predicates come from the PC ────────────────────────────────────────
 * is_grasped (_check_grasp) and check_success are CONTACT queries in MuJoCo and
 * are NOT present in the 46-float observation. Observation-only proxies were
 * built and measured against the simulator: pure proximity scored 0/8
 * placements (it fires while the fingers are still closing, tripping TEST_LIFT
 * early), and a finger-closure variant only 1-2/8, versus 93% with the true
 * checks. So the PC — which owns physics — ships both predicates in a flags
 * byte appended to the state frame, and this sketch consumes them directly.
 * The target bin position IS a fixed constant (bins don't move), so it stays
 * hardcoded here.
 *
 * HANDOFF RETRIES live on the PC. place_env_wrapper.reset() respawns and
 * retries a failed grasp/test-lift up to 8 times BEFORE scoring — the 92%
 * baseline is conditional on that loop, and only the PC can respawn. This
 * sketch reports its current phase in a status byte so hil_main.py knows when
 * the handoff succeeded (PH_TRANSPORT) and when the episode ended (PH_DONE_*),
 * and it accepts a RESET frame to resync its FSM between attempts.
 *
 * Serial protocol (v2):
 *   PC  -> MCU : [SYNC:4][type:1][seq:1][len:2][state:184][flags:1][crc:2]
 *   MCU -> PC  : [SYNC:4][type:1][seq:1][len:2][action:28][status:1][crc:2]
 */

#include <ArduTFLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Two INT8 models. Generate with:
//   python qat_and_convert.py --actor_path checkpoints/td3_grasp/actor_td3 \
//       --input_dims 46 --n_actions 7 --fc1 64 --fc2 32 \
//       --replay_buffer_path demos_bread.npz --output_path grasp_int8.tflite
//   python tflite_to_header.py grasp_int8.tflite grasp_model.h
//   (place model similarly -> place_model.h)
#include "grasp_model.h"   // grasp_model_tflite[],  grasp_model_tflite_len
#include "place_model.h"   // place_model_tflite[],  place_model_tflite_len
#include "corrector_model.h"  // FP32 AprilTag residual corrector (20.3 KB)

// ── Dimensions ─────────────────────────────────────────────────────────────
#define STATE_DIM  46
#define ACTION_DIM 7

// ── Observation indices (from GymWrapper flatten order; verified in sim) ────
//   [0:3]   Bread_pos (object XYZ)
//   [7:10]  Bread_to_robot0_eef_pos (object relative to gripper)
#define OBJ_X 0
#define OBJ_Y 1
#define OBJ_Z 2
#define O2E_X 7
#define O2E_Y 8
#define O2E_Z 9

// ── Fixed world constants (bread's target sub-bin center) ──────────────────
static const float BIN_X = 0.1975f;
static const float BIN_Y = 0.1575f;
static const float BIN_Z = 0.80f;

// ── Flag bits supplied by the PC (true MuJoCo contact queries) ─────────────
#define FLAG_GRASPED 0x01
#define FLAG_SUCCESS 0x02
static const float TABLE_Z = 0.78f;   // obj z below this = fell off table

// ── Locked deployment FSM params (place_env_wrapper.py + eval sweep) ────────
// Rule-layer values re-tuned for RANDOM spawn (Results/fsm_rule_layer_sweep.txt).
// Sweep at 1200 episodes/arm: 87.33% -> 89.58% (+2.25 pts, z=+1.73, ns).
// Previous fixed-spawn values were NEAR_TARGET_XY 0.14, TRANSLATE_SCALE 0.5,
// CARRY_GAIN 4.0. NEAR_TARGET_XY 0.10 was measured at -11 points (the release
// trigger stops firing), so do not tighten this below ~0.12.
static const float NEAR_TARGET_XY   = 0.18f; // over-bin release trigger radius
static const int   RELEASE_TRIG_HOLD = 3;    // consecutive over-bin steps
static const int   PLACE_HORIZON    = 300;   // transport-phase step cap
static const int   GRASP_HOLD       = 8;     // consecutive grasp frames to confirm
static const int   GRASP_CAP        = 250;   // give-up cap for grasp phase
// The only rule-layer parameter with a coherent trend in the sweep:
// 0.4 -> 80%, 0.5 -> 82%, 0.6 -> 86%, 0.7 -> 86%, 0.8 -> 85%, 1.0 -> 83%.
static const float TRANSLATE_SCALE  = 0.65f; // transport translation gentling

// Test-lift (grasp-robustness probe)
static const int   TL_STEPS  = 20;
static const float TL_DZ     = 0.5f;
static const float TL_MIN_RISE = 0.03f;

// Scripted release (P-controller phases)
static const float CARRY_GAIN = 6.0f;
static const float CARRY_CLIP = 0.5f;
// Fix C (Results/fixC): RC_STEPS 30 -> 60. RECENTER exits on either reaching
// RC_TOL of the bin centre or exhausting this budget; at 30 the budget was the
// binding exit for a large share of episodes, releasing the object short of
// centre. Worth +0.50 INT8 end-to-end (p=0.031, 0 of 12 seeds worse).
// Fix A (Results/fixA): lost-grip recovery in TRANSPORT. Measured on the host,
// 3.33% FP32 / 10.50% INT8 of episodes drop the object a median 9% into the
// carry and then fly an EMPTY gripper for the whole PLACE_HORIZON, because the
// release trigger requires `grasped`. TEST_LIFT already had this branch;
// TRANSPORT did not. Worth +1.75 INT8 / +0.75 FP32 (paired t=4.71, p=0.0010,
// 11 of 12 seeds up, 0 down).
static const int   LOST_GRIP_STEPS = 5;   // consecutive ungrasped frames
static const int   MAX_REGRASP     = 2;   // regrasp attempts per episode
static const int   RC_STEPS  = 60;    static const float RC_TOL   = 0.03f;
static const int   DS_STEPS  = 30;    static const float DS_DZ    = -0.12f;
static const float TOUCH_MARGIN = 0.02f;
static const int   OP_STEPS  = 8;
static const int   RT_STEPS  = 12;    static const float RT_DZ    = 0.3f;

// ── Protocol constants (identical to FP32 sketch) ──────────────────────────
#define STATE_MSG 0x01
#define ACTION_MSG 0x02
#define RESET_MSG 0x03
// Carries the 46-float observation PLUS the 12 detector features the residual
// corrector needs. The object-pose block arrives UNCORRECTED (straight from
// solvePnP) and this sketch corrects it, so part of perception runs here
// rather than on the PC.
#define CORR_MSG  0x04
#define SYNC_BYTE_0 0xAA
#define SYNC_BYTE_1 0x55
#define SYNC_BYTE_2 0xAA
#define SYNC_BYTE_3 0x55
#define STATE_PAYLOAD_SIZE 191   // type1+seq1+len2+state184+flags1+crc2
#define CORR_PAYLOAD_SIZE  239   // ... +12 feature floats (48 B)
#define CORR_FEATS 12
#define ACTION_PAYLOAD_SIZE 35   // type1+seq1+len2+action28+status1+crc2

// ── FSM states ─────────────────────────────────────────────────────────────
enum Phase {
  PH_GRASP, PH_TEST_LIFT, PH_TRANSPORT,
  PH_RECENTER, PH_DESCEND, PH_OPEN, PH_RETRACT,
  PH_DONE_OK, PH_DONE_FAIL
};

struct FSM {
  Phase phase = PH_GRASP;
  int   grasp_hold = 0;       // consecutive is_grasped frames (grasp phase)
  int   grasp_steps = 0;      // total grasp-phase steps
  int   tl_steps = 0;         // test-lift steps
  float tl_base_z = 0.0f;     // object z when test-lift began
  int   tr_steps = 0;         // transport steps
  int   over_bin = 0;         // consecutive over-bin steps
  int   ph_steps = 0;         // steps in current scripted release phase
  float prev_z = 1e9f;        // for descend z-stall detection
  int   lost = 0;             // Fix A: consecutive ungrasped frames in carry
  int   regrasps = 0;         // Fix A: regrasp attempts used this episode
} fsm;

// ── Two interpreters, two arenas ───────────────────────────────────────────
namespace {
  const tflite::Model* grasp_model = nullptr;
  const tflite::Model* place_model = nullptr;
  tflite::MicroInterpreter* grasp_interp = nullptr;
  tflite::MicroInterpreter* place_interp = nullptr;
  TfLiteTensor *grasp_in=nullptr, *grasp_out=nullptr;
  TfLiteTensor *place_in=nullptr, *place_out=nullptr;

  // INT8 64->32->7 nets are tiny; 40KB each is generous headroom.
  constexpr int kArena = 40 * 1024;
  uint8_t* grasp_arena = nullptr;
  uint8_t* place_arena = nullptr;
}

struct Stats { uint32_t total=0; uint32_t episodes=0; float avg_ms=0.0f;
               uint32_t corrections=0; float corr_ms=0.0f; } stats;

// ── Prototypes ─────────────────────────────────────────────────────────────
bool  waitForSync();
int   receiveState(float* state, uint8_t* seq, uint8_t* flags); // 1=state,3=reset,0=err
void  sendAction(const float* action, uint8_t seq, uint8_t status);
uint16_t checksum(const uint8_t* d, size_t n);
bool  setupModel(const tflite::Model** m, const unsigned char* data,
                 tflite::MicroInterpreter** it, uint8_t* arena,
                 TfLiteTensor** in, TfLiteTensor** out, const char* tag);
void  runModel(tflite::MicroInterpreter* it, TfLiteTensor* in, TfLiteTensor* out,
               const float* state, float* action);
void  computeAction(const float* s, uint8_t flags, float* a);
void  selfTest();

// ── Small helpers ──────────────────────────────────────────────────────────
static inline float clampf(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }
static inline float binXYDist(const float* s){
  float dx=s[OBJ_X]-BIN_X, dy=s[OBJ_Y]-BIN_Y;
  return sqrtf(dx*dx+dy*dy);
}
static inline void zeroAction(float* a){ for(int i=0;i<ACTION_DIM;i++) a[i]=0.0f; }
// P-control XY toward bin center (release phases): a[0],a[1].
static inline void pXYtoBin(const float* s, float* a){
  a[0]=clampf(CARRY_GAIN*(BIN_X-s[OBJ_X]), -CARRY_CLIP, CARRY_CLIP);
  a[1]=clampf(CARRY_GAIN*(BIN_Y-s[OBJ_Y]), -CARRY_CLIP, CARRY_CLIP);
}

void resetFSM(){
  fsm = FSM();
  stats.episodes++;
}

// Boot-time inference self-test. Runs a DETERMINISTIC input vector through both
// interpreters and prints the outputs, so on-device inference can be compared
// against desktop TFLite on the identical input — completely independent of the
// serial path, the FSM and the simulator. Expected values are printed by
// selftest_expected.py on the PC.
void selfTest(){
  static float tv[STATE_DIM];
  for(int i=0;i<STATE_DIM;i++) tv[i] = (i - 23) * 0.05f;   // -1.15 .. +1.10
  float a[ACTION_DIM];

  // Board capability probe. On-device AprilTag detection needs a 1280x960
  // decode buffer (1200 KB); internal SRAM leaves ~253 KB free, so PSRAM is a
  // hard requirement for that path. Print it once at boot so the answer is a
  // measurement rather than an assumption.
  Serial.printf("[board] free internal heap: %d KB\n", ESP.getFreeHeap()/1024);
#if defined(BOARD_HAS_PSRAM) || defined(CONFIG_SPIRAM_SUPPORT)
  Serial.printf("[board] PSRAM: %s, %d KB free\n",
                psramFound() ? "PRESENT" : "absent", ESP.getFreePsram()/1024);
#else
  Serial.println("[board] PSRAM: not enabled in this build "
                 "(Tools > PSRAM: Enabled, on a WROVER/S3/CAM board)");
#endif
  Serial.println("\n--- INFERENCE SELF-TEST (input: s[i]=(i-23)*0.05) ---");
  runModel(grasp_interp, grasp_in, grasp_out, tv, a);
  Serial.print("grasp:");
  for(int i=0;i<ACTION_DIM;i++) Serial.printf(" %+.6f", a[i]);
  Serial.println();

  runModel(place_interp, place_in, place_out, tv, a);
  Serial.print("place:");
  for(int i=0;i<ACTION_DIM;i++) Serial.printf(" %+.6f", a[i]);
  Serial.println();
  Serial.println("--- END SELF-TEST ---\n");
}

// ── setup ──────────────────────────────────────────────────────────────────
void setup(){
  Serial.begin(921600);
  while(!Serial) delay(10);
  Serial.println("\n===============================================");
  Serial.println("ESP32 INT8 Pick-and-Place FSM (grasp + transport)");
  Serial.println("===============================================");

  grasp_arena = (uint8_t*)malloc(kArena);
  place_arena = (uint8_t*)malloc(kArena);
  if(!grasp_arena || !place_arena){
    Serial.println("FATAL: arena malloc failed"); while(1) delay(1000);
  }

  if(!setupModel(&grasp_model, grasp_model_tflite, &grasp_interp, grasp_arena,
                 &grasp_in, &grasp_out, "grasp")) { while(1) delay(1000); }
  if(!setupModel(&place_model, place_model_tflite, &place_interp, place_arena,
                 &place_in, &place_out, "place")) { while(1) delay(1000); }

  Serial.printf("grasp in/out types: %d/%d  place in/out: %d/%d\n",
                grasp_in->type, grasp_out->type, place_in->type, place_out->type);
  selfTest();
  Serial.println("READY. Waiting for states...\n");
  resetFSM();
  stats.episodes = 0;
}

// ── main loop ──────────────────────────────────────────────────────────────
void loop(){
  static float state[STATE_DIM];
  static float action[ACTION_DIM];
  static uint8_t seq = 0;
  static uint8_t flags = 0;

  int r = receiveState(state, &seq, &flags);
  if(r == 3){ resetFSM(); return; }        // explicit RESET frame from the PC
  if(r != 1) return;                        // error / no state

  // Auto-reset: a state arriving after the episode ended = new episode.
  if(fsm.phase == PH_DONE_OK || fsm.phase == PH_DONE_FAIL) resetFSM();

  unsigned long t0 = micros();
  computeAction(state, flags, action);
  float ms = (micros()-t0)/1000.0f;

  stats.total++;
  stats.avg_ms = (stats.avg_ms*(stats.total-1)+ms)/stats.total;

  sendAction(action, seq, (uint8_t)fsm.phase);

  if(stats.total % 100 == 0){
    Serial.printf("[%lu] ep=%lu phase=%d avg=%.2fms corr=%.3fms n=%lu heap=%dKB\n",
                  stats.total, stats.episodes, fsm.phase, stats.avg_ms,
                  stats.corr_ms, stats.corrections, ESP.getFreeHeap()/1024);
  }
}

// ── The FSM: pick the action for this state ────────────────────────────────
void computeAction(const float* s, uint8_t flags, float* a){
  zeroAction(a);
  bool grasped = (flags & FLAG_GRASPED) != 0;   // true MuJoCo contact check
  bool placed  = (flags & FLAG_SUCCESS) != 0;   // true _check_success()
  bool over    = binXYDist(s) <= NEAR_TARGET_XY;

  // Fell off the table at any point after grasp: fail out.
  if(fsm.phase != PH_GRASP && s[OBJ_Z] < TABLE_Z){
    fsm.phase = PH_DONE_FAIL;
  }

  switch(fsm.phase){

    // ── GRASP: raw grasp model controls all 7 dims (incl. gripper) ─────────
    case PH_GRASP: {
      runModel(grasp_interp, grasp_in, grasp_out, s, a);   // fills a[0..6]
      fsm.grasp_steps++;
      if(grasped){
        if(++fsm.grasp_hold >= GRASP_HOLD){
          fsm.phase = PH_TEST_LIFT; fsm.tl_steps = 0; fsm.tl_base_z = s[OBJ_Z];
        }
      } else fsm.grasp_hold = 0;
      if(fsm.grasp_steps >= GRASP_CAP) fsm.phase = PH_DONE_FAIL;
      break;
    }

    // ── TEST_LIFT: scripted straight-up probe, gripper closed (raw) ────────
    case PH_TEST_LIFT: {
      a[2] = TL_DZ; a[6] = 1.0f;                 // +z, gripper closed
      fsm.tl_steps++;
      if(!grasped){                              // slipped -> retry grasp
        fsm.phase = PH_GRASP; fsm.grasp_hold = 0;  // NOTE: no respawn on MCU
        break;
      }
      if(fsm.tl_steps >= TL_STEPS){
        bool rose = (s[OBJ_Z] - fsm.tl_base_z) >= TL_MIN_RISE;
        if(rose){ fsm.phase = PH_TRANSPORT; fsm.tr_steps = 0; fsm.over_bin = 0; }
        else    { fsm.phase = PH_GRASP; fsm.grasp_hold = 0; }
      }
      break;
    }

    // ── TRANSPORT: place model, orientation frozen, translation gentled,
    //    gripper scripted closed. Trigger scripted release when parked over bin.
    case PH_TRANSPORT: {
      runModel(place_interp, place_in, place_out, s, a);
      a[3]=a[4]=a[5]=0.0f;                        // freeze orientation
      a[0]*=TRANSLATE_SCALE; a[1]*=TRANSLATE_SCALE; a[2]*=TRANSLATE_SCALE;
      a[6]=1.0f;                                  // gripper scripted closed
      fsm.tr_steps++;
      fsm.over_bin = over ? fsm.over_bin+1 : 0;
      fsm.lost = grasped ? 0 : fsm.lost + 1;      // Fix A
      if(grasped && fsm.over_bin >= RELEASE_TRIG_HOLD){
        fsm.phase = PH_RECENTER; fsm.ph_steps = 0;
      } else if(fsm.lost >= LOST_GRIP_STEPS && fsm.regrasps < MAX_REGRASP){
        // Fix A: the object was dropped mid-carry. Go back and pick it up
        // instead of flying an empty gripper to the horizon.
        fsm.regrasps++;
        fsm.phase = PH_GRASP; fsm.grasp_hold = 0; fsm.grasp_steps = 0;
        fsm.lost = 0;
      } else if(fsm.tr_steps >= PLACE_HORIZON){
        fsm.phase = PH_DONE_FAIL;                 // never delivered to bin
      }
      break;
    }

    // ── RECENTER: P toward bin center, hold height, gripper closed ─────────
    case PH_RECENTER: {
      pXYtoBin(s, a); a[6]=1.0f;
      fsm.ph_steps++;
      if(binXYDist(s) <= RC_TOL || fsm.ph_steps >= RC_STEPS || !grasped){
        fsm.phase = PH_DESCEND; fsm.ph_steps = 0; fsm.prev_z = 1e9f;
      }
      break;
    }

    // ── DESCEND: keep XY centered, command gentle -z until touchdown/stall ─
    case PH_DESCEND: {
      pXYtoBin(s, a); a[2]=DS_DZ; a[6]=1.0f;
      fsm.ph_steps++;
      bool touched = s[OBJ_Z] <= BIN_Z + TOUCH_MARGIN;
      bool stalled = s[OBJ_Z] >= fsm.prev_z - 1e-4f;   // stopped descending
      fsm.prev_z = s[OBJ_Z];
      if(!grasped || touched || stalled || fsm.ph_steps >= DS_STEPS){
        fsm.phase = PH_OPEN; fsm.ph_steps = 0;
      }
      break;
    }

    // ── OPEN: release gripper, no arm motion, let it settle ────────────────
    case PH_OPEN: {
      a[6] = -1.0f;
      if(++fsm.ph_steps >= OP_STEPS){ fsm.phase = PH_RETRACT; fsm.ph_steps = 0; }
      break;
    }

    // ── RETRACT: lift empty gripper clear, then judge success ──────────────
    // _check_success needs the gripper ~4.2cm clear of the object, so success
    // is judged only as the gripper retracts (the PC re-evaluates each step).
    case PH_RETRACT: {
      a[2]=RT_DZ; a[6]=-1.0f;
      fsm.ph_steps++;
      if(placed) fsm.phase = PH_DONE_OK;
      else if(fsm.ph_steps >= RT_STEPS) fsm.phase = PH_DONE_FAIL;
      break;
    }

    // ── DONE: hold still (open, no motion) until the PC resets the episode ─
    case PH_DONE_OK:
    case PH_DONE_FAIL:
    default:
      a[6] = -1.0f;   // gripper open, arm still
      break;
  }

  for(int i=0;i<ACTION_DIM;i++) a[i]=clampf(a[i], -1.0f, 1.0f);
}

// ── Model setup / inference ────────────────────────────────────────────────
bool setupModel(const tflite::Model** m, const unsigned char* data,
                tflite::MicroInterpreter** it, uint8_t* arena,
                TfLiteTensor** in, TfLiteTensor** out, const char* tag){
  *m = tflite::GetModel(data);
  if((*m)->version() != TFLITE_SCHEMA_VERSION){
    Serial.printf("[%s] schema mismatch\n", tag); return false;
  }
  static tflite::AllOpsResolver resolver;   // shared resolver is fine
  // Each model needs its own persistent interpreter object.
  tflite::MicroInterpreter* interp =
    new tflite::MicroInterpreter(*m, resolver, arena, kArena);
  if(interp->AllocateTensors() != kTfLiteOk){
    Serial.printf("[%s] AllocateTensors failed (arena too small?)\n", tag);
    return false;
  }
  *it = interp;
  *in = interp->input(0);
  *out = interp->output(0);
  if((*in)->dims->data[1] != STATE_DIM || (*out)->dims->data[1] != ACTION_DIM){
    Serial.printf("[%s] bad shapes: in[1]=%d out[1]=%d\n",
                  tag, (*in)->dims->data[1], (*out)->dims->data[1]);
    return false;
  }
  Serial.printf("[%s] arena used: %d bytes\n", tag, (int)interp->arena_used_bytes());
  return true;
}

// Handles both float32-I/O and int8-I/O tflite exports (qat_and_convert.py
// defaults to float32 I/O; if you re-export with int8 I/O this still works).
void runModel(tflite::MicroInterpreter* it, TfLiteTensor* in, TfLiteTensor* out,
              const float* state, float* action){
  if(in->type == kTfLiteInt8){
    float sc = in->params.scale; int zp = in->params.zero_point;
    for(int i=0;i<STATE_DIM;i++){
      int q = (int)lroundf(state[i]/sc) + zp;
      in->data.int8[i] = (int8_t)(q<-128?-128:(q>127?127:q));
    }
  } else {
    for(int i=0;i<STATE_DIM;i++) in->data.f[i] = state[i];
  }

  if(it->Invoke() != kTfLiteOk){ zeroAction(action); return; }

  if(out->type == kTfLiteInt8){
    float sc = out->params.scale; int zp = out->params.zero_point;
    for(int i=0;i<ACTION_DIM;i++)
      action[i] = (out->data.int8[i]-zp)*sc;
  } else {
    for(int i=0;i<ACTION_DIM;i++) action[i] = out->data.f[i];
  }
}

// ── AprilTag residual corrector (FP32, hand-rolled) ────────────────────────
// 12 -> 64 -> 64 -> 3, ReLU, linear output in METRES. Deliberately not INT8:
// per-tensor quantisation was measured at 6.4-13.2 mm of error against the
// 11.6 mm residual it corrects, because the 12 features span a 2198x range of
// scales (Results/corrector_on_device.txt). No TFLite interpreter and no third
// arena -- 5,187 floats and a few thousand MACs on a core already 99.5% idle.
static void correctorForward(const float* feat, float* out3){
  float h1[CORR_H1], h2[CORR_H2];
  for(int j=0;j<CORR_H1;j++){
    float a = CORR_B0[j];
    for(int i=0;i<CORR_IN;i++)
      a += CORR_W0[j*CORR_IN+i] * ((feat[i] - CORR_MU[i]) / CORR_SD[i]);
    h1[j] = a > 0.0f ? a : 0.0f;
  }
  for(int j=0;j<CORR_H2;j++){
    float a = CORR_B1[j];
    for(int i=0;i<CORR_H1;i++) a += CORR_W1[j*CORR_H1+i] * h1[i];
    h2[j] = a > 0.0f ? a : 0.0f;
  }
  for(int j=0;j<CORR_OUT;j++){
    float a = CORR_B2[j];
    for(int i=0;i<CORR_H2;i++) a += CORR_W2[j*CORR_H2+i] * h2[i];
    out3[j] = a;
  }
}

// Apply the correction in place. Only POSITION is corrected -- the model has a
// 3-vector output -- so obj_quat and the relative quaternion are untouched, but
// obj_to_eef_pos must be re-derived because it depends on the corrected pose.
// robosuite defines it as R_eef^T (obj_pos - eef_pos); that composition is
// exact, which is why holding a world pose and recomputing the relative block
// is sound (see perception_wrapper mode="recompute").
static void applyCorrection(float* s, const float* feat){
  float d[3]; correctorForward(feat, d);
  s[0]+=d[0]; s[1]+=d[1]; s[2]+=d[2];                 // obj_pos, world
  const float* q = &s[38];                            // eef_quat, xyzw
  const float x=q[0],y=q[1],z=q[2],w=q[3];
  // R_eef columns; we need R^T v, i.e. dot of v with each column
  const float r[9] = {
    1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),
    2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),
    2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)};
  const float v[3] = { s[0]-s[35], s[1]-s[36], s[2]-s[37] };
  for(int c=0;c<3;c++)
    s[7+c] = r[0*3+c]*v[0] + r[1*3+c]*v[1] + r[2*3+c]*v[2];
}

// ── Serial protocol (mirrors pick_and_place_FP32.ino) ──────────────────────
bool waitForSync(){
  uint8_t st=0; unsigned long t=millis();
  while(millis()-t < 5000){
    if(Serial.available()>0){
      uint8_t b=Serial.read();
      if(st==0 && b==SYNC_BYTE_0) st=1;
      else if(st==1 && b==SYNC_BYTE_1) st=2;
      else if(st==2 && b==SYNC_BYTE_2) st=3;
      else if(st==3 && b==SYNC_BYTE_3) return true;
      else st = (b==SYNC_BYTE_0)?1:0;
    }
  }
  return false;
}

int receiveState(float* state, uint8_t* seq, uint8_t* flags){
  if(!waitForSync()) return 0;
  static uint8_t buf[CORR_PAYLOAD_SIZE];       // the larger of the two
  unsigned long t=millis(); size_t n=0;
  // Read the 4-byte header first so the payload size can follow the TYPE.
  // The old code read a fixed STATE_PAYLOAD_SIZE and inspected the type
  // afterwards, which cannot accommodate two frame sizes.
  while(n<4 && millis()-t<1000){ if(Serial.available()>0) buf[n++]=Serial.read(); }
  if(n<4) return 0;
  uint8_t type=buf[0]; *seq=buf[1];
  if(type==RESET_MSG){                          // drain the rest of the frame
    while(n<STATE_PAYLOAD_SIZE && millis()-t<1000)
      if(Serial.available()>0) buf[n++]=Serial.read();
    return 3;
  }
  size_t want = (type==CORR_MSG) ? CORR_PAYLOAD_SIZE : STATE_PAYLOAD_SIZE;
  if(type!=STATE_MSG && type!=CORR_MSG) return 0;
  while(n<want && millis()-t<1000){ if(Serial.available()>0) buf[n++]=Serial.read(); }
  if(n!=want) return 0;
  uint16_t rc = buf[want-2] | (buf[want-1]<<8);
  if(rc != checksum(buf, want-2)) return 0;
  memcpy(state, &buf[4], STATE_DIM*sizeof(float));
  *flags = buf[4 + STATE_DIM*sizeof(float)];   // true grasp/success from the PC
  if(type==CORR_MSG){
    float feat[CORR_FEATS];
    memcpy(feat, &buf[5 + STATE_DIM*sizeof(float)], CORR_FEATS*sizeof(float));
    unsigned long c0 = micros();
    applyCorrection(state, feat);              // perception, ON DEVICE
    float cms = (micros()-c0)/1000.0f;
    stats.corrections++;
    stats.corr_ms = (stats.corr_ms*(stats.corrections-1)+cms)/stats.corrections;
  }
  return 1;
}

void sendAction(const float* action, uint8_t seq, uint8_t status){
  uint8_t buf[ACTION_PAYLOAD_SIZE];
  buf[0]=ACTION_MSG; buf[1]=seq;
  uint16_t len=ACTION_DIM*sizeof(float)+1;
  buf[2]=len&0xFF; buf[3]=(len>>8)&0xFF;
  memcpy(&buf[4], action, ACTION_DIM*sizeof(float));
  buf[4 + ACTION_DIM*sizeof(float)] = status;  // FSM phase, for the PC's loop
  uint16_t crc=checksum(buf, ACTION_PAYLOAD_SIZE-2);
  buf[ACTION_PAYLOAD_SIZE-2]=crc&0xFF; buf[ACTION_PAYLOAD_SIZE-1]=(crc>>8)&0xFF;
  Serial.write(SYNC_BYTE_0); Serial.write(SYNC_BYTE_1);
  Serial.write(SYNC_BYTE_2); Serial.write(SYNC_BYTE_3);
  Serial.write(buf, ACTION_PAYLOAD_SIZE);
  Serial.flush();
}

uint16_t checksum(const uint8_t* d, size_t n){
  uint32_t s=0; for(size_t i=0;i<n;i++) s+=d[i]; return s%65536;
}
