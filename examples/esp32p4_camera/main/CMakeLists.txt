idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "."
    REQUIRES
        esp_driver_cam
        esp_driver_isp
        esp_driver_ldo
        esp_driver_i2c
        esp_lcd
        esp_psram
        log
        esp_timer
        facex
)
