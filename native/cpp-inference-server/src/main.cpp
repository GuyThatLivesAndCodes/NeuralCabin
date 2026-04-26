#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "neuralcabin/server.hpp"

int main(int argc, char** argv) {
  const std::uint16_t port = (argc > 1) ? static_cast<std::uint16_t>(std::stoi(argv[1])) : 8787;
  const std::size_t workers = (argc > 2) ? static_cast<std::size_t>(std::stoul(argv[2])) : 8;

  // Replace with secure key management in production (KMS/HSM/env injection).
  std::array<std::uint8_t, 32> key{};
  for (std::size_t i = 0; i < key.size(); ++i) key[i] = static_cast<std::uint8_t>(i + 1);

  neuralcabin::InferenceServer server(port, workers, key);

  // Demo placeholder payload. Real deployments load serialized model weights.
  const std::string demo_weights = "neuralcabin-weights-v1";
  server.set_weights(std::span<const std::uint8_t>(
    reinterpret_cast<const std::uint8_t*>(demo_weights.data()),
    demo_weights.size()
  ));

  std::cout << "[server] workers=" << workers << std::endl;
  server.run();
  return 0;
}
