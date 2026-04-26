#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

#include "neuralcabin/aes_gcm.hpp"

namespace neuralcabin {

class WeightStore {
public:
  WeightStore() = default;

  void load_plain(std::span<const std::uint8_t> plain, std::span<const std::uint8_t, 32> key);
  std::vector<std::uint8_t> decrypt(std::span<const std::uint8_t, 32> key) const;
  bool empty() const;

private:
  mutable std::mutex mu_;
  Aes256GcmPayload encrypted_;
  bool has_payload_ = false;
};

} // namespace neuralcabin
