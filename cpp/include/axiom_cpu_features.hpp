#pragma once

#include <string>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

namespace axiom {

struct CPUFeatures {
    std::string architecture;
    bool has_neon;
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512cd;
    bool has_avx512bw;
    bool has_avx512dq;
    bool has_avx512vl;

    void print() const {
        std::cout << "CPU Architecture: " << architecture << "\n"
                  << "  NEON SIMD:      " << (has_neon ? "YES" : "NO") << "\n"
                  << "  AVX2:           " << (has_avx2 ? "YES" : "NO") << "\n"
                  << "  AVX-512F:       " << (has_avx512f ? "YES" : "NO") << "\n"
                  << "  AVX-512CD:      " << (has_avx512cd ? "YES" : "NO") << "\n"
                  << "  AVX-512BW:      " << (has_avx512bw ? "YES" : "NO") << "\n"
                  << "  AVX-512DQ:      " << (has_avx512dq ? "YES" : "NO") << "\n";
    }
};

inline CPUFeatures detect_cpu_features() {
    CPUFeatures feat{};
    feat.has_neon = false;
    feat.has_avx2 = false;
    feat.has_avx512f = false;
    feat.has_avx512cd = false;
    feat.has_avx512bw = false;
    feat.has_avx512dq = false;
    feat.has_avx512vl = false;

#if defined(__aarch64__) || defined(_M_ARM64)
    feat.architecture = "arm64";
    feat.has_neon = true;
#elif defined(__x86_64__) || defined(_M_X64)
    feat.architecture = "x86_64";
    unsigned int eax, ebx, ecx, edx;

    // Check max basic CPUID level
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        // ECX bit 28: AVX
        // bool has_avx = (ecx & (1 << 28)) != 0;
    }

    // Extended features (leaf 7, subleaf 0)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        // EBX bit 5: AVX2
        feat.has_avx2 = (ebx & (1 << 5)) != 0;
        // EBX bit 16: AVX512F
        feat.has_avx512f = (ebx & (1 << 16)) != 0;
        // EBX bit 28: AVX512CD
        feat.has_avx512cd = (ebx & (1 << 28)) != 0;
        // EBX bit 30: AVX512BW
        feat.has_avx512bw = (ebx & (1 << 30)) != 0;
        // EBX bit 17: AVX512DQ
        feat.has_avx512dq = (ebx & (1 << 17)) != 0;
        // EBX bit 31: AVX512VL
        feat.has_avx512vl = (ebx & (1 << 31)) != 0;
    }
#else
    feat.architecture = "unknown";
#endif

    return feat;
}

} // namespace axiom
