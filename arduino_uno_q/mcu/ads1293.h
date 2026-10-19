#pragma once
#include <Arduino.h>
#include <SPI.h>

namespace ads1293reg {
    // Operation Mode Registers
    constexpr uint8_t CONFIG                0x00; /*Main Configuration*/
    # Input Channel Selection Registers
    constexpr uint8_t FLEX_CH1_CN           0x01; /*Flex Routing Switch Control for Channel 1*/
    constexpr uint8_t FLEX_CH2_CN           0x02; /*Flex Routing Switch Control for Channel 2*/
    constexpr uint8_t FLEX_CH3_CN           0x03; /*Flex Routing Switch Control for Channel 3*/
    constexpr uint8_t FLEX_PACE_CN          0x04; /*Flex Routing Switch Control for Pace Channel*/
    constexpr uint8_t FLEX_VBAT_CN          0x05; /*Flex Routing Switch for Battery Monitoring*/
    // Lead-off Detect Control Registers
    constexpr uint8_t LOD_CN                0x06; /*Lead-Off Detect Control*/
    constexpr uint8_t LOD_EN                0x07; /*Lead-Off Detect Enable*/
    constexpr uint8_t LOD_CURRENT           0x08; /*Lead-Off Detect Current*/
    constexpr uint8_t LOD_AC_CN             0x09; /*AC Lead-Off Detect Control*/
    // Common-Mode Detection and Right-Leg Drive Feedback Control Registers
    constexpr uint8_t CMDET_EN              0x0A; /*Common-Mode Detect Enable*/
    constexpr uint8_t CMDET_CN              0x0B; /*Common-Mode Detect Control*/
    constexpr uint8_t RLD_CN                0x0C; /*Right-Leg Drive Control*/
    // Wilson Control Registers
    constexpr uint8_t WILSON_EN1            0x0D; /*Wilson Reference Input one Selection*/
    constexpr uint8_t WILSON_EN2            0x0E; /*Wilson Reference Input two Selection*/
    constexpr uint8_t WILSON_EN3            0x0F; /*Wilson Reference Input three Selection*/
    constexpr uint8_t WILSON_CN             0x10; /*Wilson Reference Control*/
    // Reference Registers
    constexpr uint8_t REF_CN                0x11; /*Internal Reference Voltage Control*/
    // OSC Control Registers
    constexpr uint8_t OSC_CN                0x12; /*Clock Source and Output Clock Control*/
    // AFE Control Registers
    constexpr uint8_t AFE_RES               0x13; /*Analog Front End Frequency and Resolution*/
    constexpr uint8_t AFE_SHDN_CN           0x14; /*Analog Front End Shutdown Control*/
    constexpr uint8_t AFE_FAULT_CN          0x15; /*Analog Front End Fault Detection Control*/
    constexpr uint8_t AFE_PACE_CN           0x17; /*Analog Pace Channel Output Routing Control*/
    // Error Status Registers
    constexpr uint8_t ERROR_LOD             0x18; /*Lead-Off Detect Error Status*/
    constexpr uint8_t ERROR_STATUS          0x19; /*Other Error Status*/
    constexpr uint8_t ERROR_RANGE1          0x1A; /*Channel 1 AFE Out-of-Range Status*/
    constexpr uint8_t ERROR_RANGE2          0x1B; /*Channel 2 AFE Out-of-Range Status*/
    constexpr uint8_t ERROR_RANGE3          0x1C; /*Channel 3 AFE Out-of-Range Status*/
    constexpr uint8_t ERROR_SYNC            0x1D; /*Synchronization Error*/
    constexpr uint8_t ERROR_MISC            0x1E; /*Miscellaneous Errors*/
    // Digital Registers
    constexpr uint8_t DIGO_STRENGTH         0x1F; /*Digital Output Drive Strength*/
    constexpr uint8_t R2_RATE               0x21; /*R2 Decimation Rate*/
    constexpr uint8_t R3_RATE_CH1           0x22; /*R3 Decimation Rate for Channel 1*/
    constexpr uint8_t R3_RATE_CH2           0x23; /*R3 Decimation Rate for Channel 2*/
    constexpr uint8_t R3_RATE_CH3           0x24; /*R3 Decimation Rate for Channel 3*/
    constexpr uint8_t R1_RATE               0x25; /*R1 Decimation Rate*/
    constexpr uint8_t DIS_EFILTER           0x26; /*ECG Filter Disable*/
    constexpr uint8_t DRDYB_SRC             0x27; /*Data Ready Pin Source*/
    constexpr uint8_t SYNCB_CN              0x28; /*SYNCB In/Out Pin Control*/
    constexpr uint8_t MASK_DRDYB            0x29; /*Optional Mask Control for DRDYB Output*/
    constexpr uint8_t MASK_ERR              0x2A; /*Mask Error on ALARMB Pin*/
    constexpr uint8_t ALARM_FILTER          0x2E; /*Digital Filter for Analog Alarm Signals*/
    constexpr uint8_t CH_CNFG               0x2F; /*Configure Channel for Loop Read Back Mode*/
    // Pace and ECG Data Read Back Registers
    constexpr uint8_t DATA_STATUS           0x30; /*ECG and Pace Data Ready Status*/
    constexpr uint8_t DATA_CH1_PACE1        0x31; /*Channel 1 Pace Data*/
    constexpr uint8_t DATA_CH1_PACE2        0x32; /*Channel 1 Pace Data*/
    constexpr uint8_t DATA_CH2_PACE1        0x33; /*Channel 2 Pace Data*/
    constexpr uint8_t DATA_CH2_PACE2        0x34; /*Channel 2 Pace Data*/
    constexpr uint8_t DATA_CH3_PACE1        0x35; /*Channel 3 Pace Data*/
    constexpr uint8_t DATA_CH3_PACE2        0x36; /*Channel 3 Pace Data*/
    constexpr uint8_t DATA_CH1_ECG1         0x37; /*Channel 1 ECG Data*/
    constexpr uint8_t DATA_CH1_ECG2         0x38; /*Channel 1 ECG Data*/
    constexpr uint8_t DATA_CH1_ECG3         0x39; /*Channel 1 ECG Data*/
    constexpr uint8_t DATA_CH2_ECG1         0x3A; /*Channel 2 ECG Data*/
    constexpr uint8_t DATA_CH2_ECG2         0x3B; /*Channel 2 ECG Data*/
    constexpr uint8_t DATA_CH2_ECG3         0x3C; /*Channel 2 ECG Data*/
    constexpr uint8_t DATA_CH3_ECG1         0x3D; /*Channel 3 ECG Data*/
    constexpr uint8_t DATA_CH3_ECG2         0x3E; /*Channel 3 ECG Data*/
    constexpr uint8_t DATA_CH3_ECG3         0x3F; /*Channel 3 ECG Data*/
    constexpr uint8_t REVID                 0x40; /*Revision ID*/
    constexpr uint8_t DATA_LOOP             0x50; /*Loop Read-Back Address*/
}

class ADS1293 {
    public:
        ADS1293(int cs, int drdy, uint32_t spiHz = 4000000):
            : cs_(cs), drdy_(drdy), settings_(spiHz, MSBFIRST, SPI_MODE0) {}
        bool begin(bool leadOffDetect);
        uint8_t readReg(uint8_t addr);
        uint8_t writeReg(uint8_t addr, uint8_t value);
        uint32_t readCh1Raw();
        uint8_t leadOffFlags();
        uint8_t revision() { return readReg(ads1293reg::REVID); }

    private:
        int cs_,
        int drdy_;
        SPISettings settings_;
};