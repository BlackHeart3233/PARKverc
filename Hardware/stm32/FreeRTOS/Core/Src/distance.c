/* distance.c */
#include "distance.h"
#include "main.h"          /* for DISTANCE_SENZOR_TRIGGER_* if you use CubeMX names */
#include "FreeRTOS.h"
#include "task.h"

/* ================= USER CONFIG =================
   Pick ONE method for TRIG pin mapping:

   A) Use CubeMX-generated names (recommended):
      DISTANCE_SENZOR_TRIGGER_GPIO_Port / DISTANCE_SENZOR_TRIGGER_Pin

   B) Or hardcode:
      #define HCSR04_TRIG_PORT GPIOB
      #define HCSR04_TRIG_PIN  GPIO_PIN_0
*/

/* If you want hardcoded TRIG pin, uncomment these and comment the CubeMX ones below. */
/* #define HCSR04_TRIG_PORT GPIOB */
/* #define HCSR04_TRIG_PIN  GPIO_PIN_0 */

#ifndef HCSR04_TRIG_PORT
#define HCSR04_TRIG_PORT DISTANCE_SENZOR_TRIGGER_GPIO_Port
#endif

#ifndef HCSR04_TRIG_PIN
#define HCSR04_TRIG_PIN  DISTANCE_SENZOR_TRIGGER_Pin
#endif

/* ================= INTERNAL STATE ================= */

static TIM_HandleTypeDef *s_tim = NULL;

static uint32_t s_ic_rising  = 0;
static uint32_t s_ic_falling = 0;
static uint8_t  s_ic_state   = 0;   /* 0 = wait rising, 1 = wait falling */

static volatile float   s_distance_cm = 0.0f;
static volatile uint8_t s_has_measurement = 0;

/* Task to notify when measurement is done (optional) */
static TaskHandle_t s_notify_task = NULL;

/* ================= INTERNAL HELPERS ================= */

static inline void trigger_pulse_10us(void)
{
    HAL_GPIO_WritePin(HCSR04_TRIG_PORT, HCSR04_TRIG_PIN, GPIO_PIN_SET);

    /* Small busy wait is fine in task context.
       This is ~10us-ish depending on clock/optimization.
       For exact timing you can use a timer-based delay later. */
    for (volatile uint32_t i = 0; i < 300; i++) { __NOP(); }

    HAL_GPIO_WritePin(HCSR04_TRIG_PORT, HCSR04_TRIG_PIN, GPIO_PIN_RESET);
}

/* ================= PUBLIC API ================= */

void HCSR04_Init(TIM_HandleTypeDef *htim)
{
    s_tim = htim;

    s_ic_rising = 0;
    s_ic_falling = 0;
    s_ic_state = 0;

    s_distance_cm = 0.0f;
    s_has_measurement = 0;

    /* Start input capture interrupt on CH1 (echo input) */
    (void)HAL_TIM_IC_Start_IT(s_tim, TIM_CHANNEL_1);

    /* Ensure we start by capturing rising edge */
    __HAL_TIM_SET_CAPTUREPOLARITY(s_tim, TIM_CHANNEL_1, TIM_INPUTCHANNELPOLARITY_RISING);
}

void HCSR04_SetNotifyTaskHandle(void *taskHandle)
{
    /* CMSIS-RTOS2 task handles are compatible with FreeRTOS TaskHandle_t for notifications in this setup.
       Pass: (void*)xTaskGetCurrentTaskHandle() */
    s_notify_task = (TaskHandle_t)taskHandle;
}

void HCSR04_Trigger(void)
{
    /* Must be called from task context (not ISR) */
    if (s_tim == NULL)
        return;

    trigger_pulse_10us();
}

float HCSR04_GetDistanceCm(void)
{
    return s_distance_cm;
}

uint8_t HCSR04_HasMeasurement(void)
{
    return s_has_measurement;
}

/* ================= ISR CALLBACK ================= */

void HCSR04_TIM_IC_Callback(TIM_HandleTypeDef *htim)
{
    if (s_tim == NULL)
        return;

    /* Only react to our timer instance */
    if (htim->Instance != s_tim->Instance)
        return;

    /* Optional: only react to CH1 active */
    if (htim->Channel != HAL_TIM_ACTIVE_CHANNEL_1)
        return;

    if (s_ic_state == 0)
    {
        /* Rising edge captured */
        s_ic_rising = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
        s_ic_state  = 1;

        __HAL_TIM_SET_CAPTUREPOLARITY(htim, TIM_CHANNEL_1, TIM_INPUTCHANNELPOLARITY_FALLING);
    }
    else
    {
        /* Falling edge captured */
        s_ic_falling = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);

        uint32_t diff;
        if (s_ic_falling >= s_ic_rising)
            diff = s_ic_falling - s_ic_rising;
        else
            diff = (0xFFFFFFFFu - s_ic_rising) + s_ic_falling + 1u;

        /* Convert pulse width to cm.
           Assumes timer tick is 1us (prescaler configured accordingly).
           HC-SR04: distance_cm ≈ pulse_us / 58.0 */
        s_distance_cm = (float)diff / 58.0f;
        s_has_measurement = 1;

        s_ic_state = 0;
        __HAL_TIM_SET_CAPTUREPOLARITY(htim, TIM_CHANNEL_1, TIM_INPUTCHANNELPOLARITY_RISING);

        /* Notify task that a new measurement is ready */
        if (s_notify_task != NULL)
        {
            BaseType_t xHigherPriorityTaskWoken = pdFALSE;
            vTaskNotifyGiveFromISR(s_notify_task, &xHigherPriorityTaskWoken);
            portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
        }
    }
}
