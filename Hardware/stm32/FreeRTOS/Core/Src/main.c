/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "distance.h"
#include "usbd_cdc_if.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "queue.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim5;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* USER CODE BEGIN PV */

volatile int32_t current_pos = 0;
#define DISTANCE_MOCK   1   //1 = mock, 0 = real senzor
volatile int32_t target_pos  = 0;

osThreadId_t distanceTaskHandle;
osThreadId_t txTaskHandle;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM1_Init(void);
void StartDefaultTask(void *argument);

/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

#define RX_BUF_SIZE 64
#define STEPS_DEADBAND 3

volatile char rxBuf[RX_BUF_SIZE];
volatile uint8_t rxLen = 0;



TaskHandle_t RxTaskHandle = NULL;


typedef enum
{
    DATA_DISTANCE,
    DATA_ROTARY
} DataType_t;

typedef struct
{
    DataType_t type; //pove ali gre za razdaljo ali rotacijo
    float value; //vrednost senzorja
} TxMessage_t;

typedef struct
{
    int32_t target_position;   //želeni položaj motorja
} StepperCmd_t;


QueueHandle_t queueRotarySensor;
QueueHandle_t distanceQueue;
QueueHandle_t Queue_CMD_Stepper;
QueueHandle_t Queue_Tx;


void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance != TIM1) return;

    if (current_pos == target_pos)
    {
        HAL_TIM_Base_Stop_IT(&htim1);   // STOP TIMER = STOP MOTOR
        return;
    }

    if (current_pos < target_pos)
    {
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7, GPIO_PIN_SET); //DIR
        current_pos++;
    }
    else
    {
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7, GPIO_PIN_RESET); //DIR
        current_pos--;
    }

    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_8, GPIO_PIN_SET);
    for (volatile int i = 0; i < 50; i++);
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_8, GPIO_PIN_RESET);
}



int32_t deg_to_steps(float deg)
{
    return (int32_t)(deg * 200.0f / 360.0f);
}


void Task_Distance(void *argument)
{
#if DISTANCE_MOCK
    float mock_dist = 20.0f;
    float dir = 1.0f;

    for (;;)
    {
        // simulacija razdalje 10–80 cm
        mock_dist += dir * 0.5f;
        if (mock_dist > 80.0f) dir = -1.0f;
        if (mock_dist < 10.0f) dir =  1.0f;

        xQueueOverwrite(distanceQueue, &mock_dist);

        vTaskDelay(pdMS_TO_TICKS(60));
    }

#else
    HCSR04_SetNotifyTaskHandle(xTaskGetCurrentTaskHandle());

    for (;;)
    {
        HCSR04_Trigger();

        if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(30)) > 0)
        {
            float dist = HCSR04_GetDistanceCm();
            xQueueOverwrite(distanceQueue, &dist);
        }

        vTaskDelay(pdMS_TO_TICKS(60));
    }
#endif
}




void Task_RotarySensor  (void *argument) {
    float angle = 0.0;

    for (;;)
    {
        angle += 1.0;
        if (angle > 540) angle = -540;

        xQueueOverwrite(queueRotarySensor, &angle);
        vTaskDelay(pdMS_TO_TICKS(20));
    }


}

TaskHandle_t StepperControlHandle = NULL;
void Task_StepperControl(void *pvParameters)
{
    StepperCmd_t cmd;

    for (;;)
    {
    	xQueueReceive(Queue_CMD_Stepper, &cmd, portMAX_DELAY);
        target_pos = cmd.target_position;
        HAL_TIM_Base_Stop_IT(&htim1);
        __HAL_TIM_SET_COUNTER(&htim1, 0);
        HAL_TIM_Base_Start_IT(&htim1);
    }
}


void Task_Tx(void *argument)
{
    float last_distance = 0.0f;
    float last_rotary   = 0.0f;
    uint8_t have_dist = 0;
    uint8_t have_rot  = 0;

    char buf[64];

    for (;;)
    {
        /* Prejmi novo ROTARY vrednost */
        if (xQueueReceive(queueRotarySensor, &last_rotary, 0) == pdTRUE)
        {
            have_rot = 1;
        }

        /* Prejmi novo DISTANCE vrednost */
        if (xQueueReceive(distanceQueue, &last_distance, 0) == pdTRUE)
        {
            have_dist = 1;
        }

        /* Če imamo OBA nova podatka → pošlji */
        if (have_rot && have_dist)
        {
            int len = snprintf(buf, sizeof(buf),
                "ROT:%.2f DIST:%.2f\r\n",
                last_rotary,
                last_distance);

            while (CDC_Transmit_FS((uint8_t *)buf, len) == USBD_BUSY)
            {
                vTaskDelay(pdMS_TO_TICKS(2));
            }

            /* reset flagov – čakamo na NOV frame */
            have_rot  = 0;
            have_dist = 0;
        }

        vTaskDelay(pdMS_TO_TICKS(5));
    }
}





void Task_Rx(void *pvParameters)
{
    uint32_t notif;
    StepperCmd_t cmd;

    for (;;)
    {
        xTaskNotifyWait(0, 0xFFFFFFFF, &notif, portMAX_DELAY);

        if (strncmp((char *)rxBuf, "ROTATE:", 7) == 0)
        {
            float deg = atof((char *)&rxBuf[7]);
            int32_t new_target = deg_to_steps(deg);

            if (abs(new_target - target_pos) < STEPS_DEADBAND)
                continue;

            cmd.target_position = new_target;
            xQueueOverwrite(Queue_CMD_Stepper, &cmd);
        }
    }
}







const osThreadAttr_t distanceTask_attributes = {
  .name = "DistanceTask",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};

const osThreadAttr_t txTask_attributes = {
  .name = "TxTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityAboveNormal,
};


const osThreadAttr_t StepperTask_attributes  = {
		  .name = "Stepper",
		  .stack_size = 256 * 4,
		  .priority = (osPriority_t) osPriorityAboveNormal,
};

const osThreadAttr_t RxTask_attributes  = {
		  .name = "Task_Rx",
		  .stack_size = 256 * 4,
		  .priority = (osPriority_t) osPriorityAboveNormal,
};

const osThreadAttr_t rotaryTask_attributes = {
  .name = "RotaryTask",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};





/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM5_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */
  MX_USB_DEVICE_Init();
  HCSR04_Init(&htim5);

  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  queueRotarySensor = xQueueCreate(1, sizeof(float));
  configASSERT(queueRotarySensor);

  distanceQueue = xQueueCreate(1, sizeof(float));
  configASSERT(distanceQueue);

  Queue_CMD_Stepper = xQueueCreate(1, sizeof(StepperCmd_t));
  configASSERT(Queue_CMD_Stepper);

  Queue_Tx = xQueueCreate(8, sizeof(TxMessage_t));
  configASSERT(Queue_Tx);

  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  osThreadNew(Task_Distance, NULL, &distanceTask_attributes);
  osThreadNew(Task_Tx, NULL, &txTask_attributes);
  osThreadNew(Task_StepperControl, NULL, &StepperTask_attributes);
  RxTaskHandle = osThreadNew(Task_Rx, NULL, &RxTask_attributes);
  osThreadNew(Task_RotarySensor, NULL, &rotaryTask_attributes);
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState       = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSI;

  /* PLL delilniki – pusti enake kot prej za test */
  RCC_OscInitStruct.PLL.PLLM = 16;   // 16 MHz / 16 = 1 MHz
  RCC_OscInitStruct.PLL.PLLN = 192;  // 1 MHz * 192 = 192 MHz
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV6; // 192 / 6 = 32 MHz SYSCLK
  RCC_OscInitStruct.PLL.PLLQ = 4;    // 192 / 4 = 48 MHz za USB

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                              | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enables the Clock Security System
  */
  HAL_RCC_EnableCSS();
}


/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 1399;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 99;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_IC_InitTypeDef sConfigIC = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 59;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 4294967295;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim5, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_IC_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_RISING;
  sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;
  sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;
  sConfigIC.ICFilter = 0;
  if (HAL_TIM_IC_ConfigChannel(&htim5, &sConfigIC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(DISTANCE_SENZOR_TRIGGER_GPIO_Port, DISTANCE_SENZOR_TRIGGER_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_13|GPIO_PIN_14, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_7|GPIO_PIN_8, GPIO_PIN_RESET);

  /*Configure GPIO pin : DISTANCE_SENZOR_TRIGGER_Pin */
  GPIO_InitStruct.Pin = DISTANCE_SENZOR_TRIGGER_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(DISTANCE_SENZOR_TRIGGER_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PD13 PD14 */
  GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : PC7 PC8 */
  GPIO_InitStruct.Pin = GPIO_PIN_7|GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* init code for USB_DEVICE */
  /* USER CODE BEGIN 5 */
  /* Infinite loop */
  for(;;)
  {
    osDelay(1);
  }
  /* USER CODE END 5 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
