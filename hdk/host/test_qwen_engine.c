// Host program for the Qwen int8 matvec engine on a real f2 FPGA. Peeks/pokes the
// OCL AXI-Lite BAR of the loaded AFI: writes the activation, per-row dequant
// multipliers, and the weight tile, pulses start, polls done, reads the int16
// results, and checks them against the golden dequant matvec computed on the host.
// This is the same test sim_engine runs, but over PCIe against silicon.
//
// Build on an f2 with the AWS FPGA SDK sourced:
//   gcc test_qwen_engine.c -o test_qwen_engine -I$SDK_DIR/userspace/include \
//       -lfpga_mgmt -lfpga_pci -lrt
// Run (after fpga-load-local-image -S 0 -I <agfi>):  sudo ./test_qwen_engine

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

#define OCL_REGS 0x000000
#define OCL_X    0x010000
#define OCL_MULT 0x020000
#define OCL_Y    0x080000
#define OCL_W    0x100000
#define R_CTRL   0x00
#define R_STAT   0x04
#define R_IN     0x08
#define R_OUT    0x0C

static const int IN = 1024, OUT = 64;

int main(int argc, char** argv) {
    int slot = (argc > 1) ? atoi(argv[1]) : 0;
    int8_t  *x = malloc(IN);
    int8_t  *w = malloc((size_t)OUT * IN);
    int32_t *mult = malloc(OUT * sizeof(int32_t));
    int16_t *gold = malloc(OUT * sizeof(int16_t));

    for (int i = 0; i < IN; i++)  x[i] = (int8_t)(((i*7+3)%127)-63);
    for (int o = 0; o < OUT; o++) { for (int i = 0; i < IN; i++) w[(size_t)o*IN+i] = (int8_t)(((o*13+i*5+1)%127)-63); mult[o] = 200+o; }
    for (int o = 0; o < OUT; o++) { long long acc=0; for (int i=0;i<IN;i++) acc += (long long)x[i]*w[(size_t)o*IN+i];
        long long p = acc*(long long)mult[o] + (1LL<<15); gold[o] = (int16_t)(p>>16); }

    if (fpga_mgmt_init()) { printf("fpga_mgmt_init failed\n"); return 1; }
    pci_bar_handle_t h = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &h)) { printf("attach failed (is the AFI loaded on slot %d?)\n", slot); return 1; }

    // load dims, activation, multipliers, weights
    fpga_pci_poke(h, OCL_REGS+R_IN,  IN);
    fpga_pci_poke(h, OCL_REGS+R_OUT, OUT);
    for (int i = 0; i < IN; i++)  fpga_pci_poke(h, OCL_X + i*4, (uint8_t)x[i]);
    for (int o = 0; o < OUT; o++) fpga_pci_poke(h, OCL_MULT + o*4, (uint32_t)mult[o]);
    for (int o = 0; o < OUT; o++) for (int i = 0; i < IN; i++)
        fpga_pci_poke(h, OCL_W + ((size_t)o*IN+i)*4, (uint8_t)w[(size_t)o*IN+i]);

    // run
    fpga_pci_poke(h, OCL_REGS+R_CTRL, 1);
    uint32_t st = 0; long spins = 0;
    do { fpga_pci_peek(h, OCL_REGS+R_STAT, &st); } while (!(st & 1) && ++spins < 100000000);

    // read + check
    int bad = 0; int16_t y0 = 0;
    for (int o = 0; o < OUT; o++) { uint32_t v; fpga_pci_peek(h, OCL_Y + o*4, &v);
        int16_t y = (int16_t)(v & 0xFFFF); if (o==0) y0=y; if (y != gold[o]) bad++; }

    printf("FPGA qwen_matvec_engine: %d/%d rows correct  (y[0]=%d golden=%d, status=0x%x)\n",
           OUT-bad, OUT, y0, gold[0], st);
    printf("%s\n", bad==0 ? "PASS  int8 matvec ran on the FPGA and matches golden" : "FAIL");
    fpga_pci_detach(h);
    return bad==0 ? 0 : 1;
}
