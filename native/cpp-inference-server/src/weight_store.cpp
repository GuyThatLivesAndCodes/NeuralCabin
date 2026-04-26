#include "neuralcabin/weight_store.hpp"

#include <stdexcept>

namespace neuralcabin {

void WeightStore::load_plain(std::span<const std::uint8_t> plain, std::span<const std::uint8_t, 32> key) {
  std::lock_guard<std::mutex> lock(mu_);
  encrypted_ = encrypt_aes256_gcm(plain, key);
  has_payload_ = true;
}

std::vector<std::uint8_t> WeightStore::decrypt(std::span<const std::uint8_t, 32> key) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (!has_payload_) {
    throw std::runtime_error("weights are not loaded");
  }
  return decrypt_aes256_gcm(encrypted_, key);
}

bool WeightStore::empty() const {
  std::lock_guard<std::mutex> lock(mu_);
  return !has_payload_;
}

} // namespace neuralcabin
