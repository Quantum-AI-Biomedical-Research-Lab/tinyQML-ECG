#include "ads1293.h"

using namespace ads1293reg;

uint8_t ADS1293::readReg(uint8_t addr) {
    SPI.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    SPI.transfer(0x80 | addr);
    uint8_t v = SPI.transfer(0x00);
    digitalWrite(cs_, HIGH);
    SPI.endTransaction();
    return v;
}

void ADS1293::writeReg(uint8_t addr, uint8_t value) {
    SPI.beginTransaction(settings_);
    digitalWrite(cs_, LOW);
    SPI.transfer(addr & 0x7F);
    SPI.transfer(value);
    digitalWrite(cs_, HIGH);
    SPI.endTransaction();
}

uint32_t ADS1293::readCh1Raw() {
    const uint32_t b2 = readReg(DATA_CH1_ECG);
    const uint32_t b1 = readReg(DATA_CH1_ECG + 1);
    const uint32_t b0 = readReg(DATA_CH1_ECG + 2);
    return (b2 << 16) | (b1 << 8) | b0;
}

uint8_t ADS1293::leadOffFlags() {
    return readReg(ERR_LOD) & 0x3F;
}

bool ADS1293::begin(bool leadOffDetect) {
    pinMode(cs_, OUTPUT);
    digitalWrite(cs_, HIGH);
    pinMode(drdy_, INPUT);
    SPI.begin();
    delay(20);
    const uint8_t rev = revision();
    if (rev == 0x00 || rev == 0xFF) return false;
    writeReg(CONFIG, 0x00);
    writeReg(FLEX_CH1_CN, 0x11);
    writeReg(FLEX_CH2_CN, 0x00);
    writeReg(FLEX_CH3_CN, 0x00);
    writeReg(CMDET_EN, 0x03);
    writeReg(RLD_CN, 0x04);
    writeReg(OSC_CN, 0x04);
    writeReg(AFE_SHDN_CN, 0x36);
    writeReg(R2_RATE, 0x02);
    writeReg(R3_RATE_CH1, 0x02);
    writeReg(DRDYB_SRC, 0x08);
    writeReg(CH_CNFG, 0x10);
    if (leadOffDetect) writeReg(LOD_EN, 0x03);
    writeReg(CONFIG, 0x01);
    return true;
}