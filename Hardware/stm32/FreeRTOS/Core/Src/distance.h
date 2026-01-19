/*
 * distance.h
 *
 *  Created on: Jan 11, 2026
 *      Author: obrez
 */

#ifndef DISTANCE_H
#define DISTANCE_H

#include "stm32f4xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

void HCSR04_Init(TIM_HandleTypeDef *htim);

void HCSR04_SetNotifyTaskHandle(void *taskHandle);

void HCSR04_Trigger(void);

float HCSR04_GetDistanceCm(void);

uint8_t HCSR04_HasMeasurement(void);

void HCSR04_TIM_IC_Callback(TIM_HandleTypeDef *htim);

#ifdef __cplusplus
}
#endif

#endif /* DISTANCE_H */
