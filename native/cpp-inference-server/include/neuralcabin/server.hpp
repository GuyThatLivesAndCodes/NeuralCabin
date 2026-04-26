#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/thread_pool.hpp>

#include "neuralcabin/weight_store.hpp"

namespace neuralcabin {

class InferenceServer {
public:
  InferenceServer(
    std::uint16_t port,
    std::size_t worker_threads,
    std::array<std::uint8_t, 32> weight_key
  );

  void set_weights(std::span<const std::uint8_t> plain_weights);
  void run();
  void stop();

private:
  void handle_client(boost::asio::ip::tcp::socket socket);
  std::string run_inference(const std::string& request_json);

  std::uint16_t port_;
  boost::asio::thread_pool workers_;
  std::atomic<bool> running_{true};
  std::array<std::uint8_t, 32> weight_key_{};
  WeightStore store_;
};

} // namespace neuralcabin
