#ifndef CAR_GLOBAL_RF_MODEL_H
#define CAR_GLOBAL_RF_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define CAR_RF_FEATURE_COUNT 65u
#define CAR_RF_CLASS_COUNT 11u
#define CAR_RF_TREE_COUNT 128u
#define CAR_RF_NODE_COUNT 39588u

/* Class indices map to these labels, in order: */
/* 0: braking__worn_out_brakes */
#define CAR_RF_CLASS_BRAKING_WORN_OUT_BRAKES 0u
/* 1: idle__combined__no oil_serpentine belt */
#define CAR_RF_CLASS_IDLE_COMBINED_NO_OIL_SERPENTINE_BELT 1u
/* 2: idle__combined__power steering combined_no oil */
#define CAR_RF_CLASS_IDLE_COMBINED_POWER_STEERING_COMBINED_NO_OIL 2u
/* 3: idle__combined__power steering combined_no oil_serpentine belt */
#define CAR_RF_CLASS_IDLE_COMBINED_POWER_STEERING_COMBINED_NO_OIL_SERPENTINE_BELT 3u
/* 4: idle__combined__power steering combined_serpentine belt */
#define CAR_RF_CLASS_IDLE_COMBINED_POWER_STEERING_COMBINED_SERPENTINE_BELT 4u
/* 5: idle__low_oil */
#define CAR_RF_CLASS_IDLE_LOW_OIL 5u
/* 6: idle__power_steering */
#define CAR_RF_CLASS_IDLE_POWER_STEERING 6u
/* 7: idle__serpentine_belt */
#define CAR_RF_CLASS_IDLE_SERPENTINE_BELT 7u
/* 8: normal */
#define CAR_RF_CLASS_NORMAL 8u
/* 9: startup__bad_ignition */
#define CAR_RF_CLASS_STARTUP_BAD_IGNITION 9u
/* 10: startup__dead_battery */
#define CAR_RF_CLASS_STARTUP_DEAD_BATTERY 10u

typedef struct {
    float feature_sum[CAR_RF_FEATURE_COUNT];
    uint32_t windows_seen;
} car_rf_clip_accumulator_t;

/* Add each 1-second feature vector from the clip (11 windows for 6 s at 50% overlap).
   finish averages the 65 features, then returns one diagnosis for the full clip. */
void car_rf_clip_reset(car_rf_clip_accumulator_t *accumulator);
int car_rf_clip_add_window(car_rf_clip_accumulator_t *accumulator,
                           const float features[CAR_RF_FEATURE_COUNT]);
/* Returns the class index, or -1 if no window was added. */
int car_rf_clip_finish(const car_rf_clip_accumulator_t *accumulator,
                      float mean_probabilities[CAR_RF_CLASS_COUNT]);
const char *car_rf_class_name(int class_index);

#endif /* CAR_GLOBAL_RF_MODEL_H */
