#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace neuralcabin {

struct Aes256GcmPayload {
  std::vector<std::uint8_t> ciphertext;
  std::array<std::uint8_t, 12> nonce{};
  std::array<std::uint8_t, 16> tag{};
};

Aes256GcmPayload encrypt_aes256_gcm(
  std::span<const std::uint8_t> plaintext,
  std::span<const std::uint8_t, 32> key
);

std::vector<std::uint8_t> decrypt_aes256_gcm(
  const Aes256GcmPayload& payload,
  std::span<const std::uint8_t, 32> key
);

} // namespace neuralcabin
