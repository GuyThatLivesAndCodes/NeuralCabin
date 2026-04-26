#include "neuralcabin/server.hpp"

#include <iostream>
#include <sstream>
#include <utility>

#include <boost/asio/post.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

namespace neuralcabin {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = net::ip::tcp;

InferenceServer::InferenceServer(
  std::uint16_t port,
  std::size_t worker_threads,
  std::array<std::uint8_t, 32> weight_key
)
  : port_(port),
    workers_(worker_threads == 0 ? 1 : worker_threads),
    weight_key_(weight_key) {}

void InferenceServer::set_weights(std::span<const std::uint8_t> plain_weights) {
  store_.load_plain(plain_weights, weight_key_);
}

void InferenceServer::stop() {
  running_.store(false, std::memory_order_relaxed);
}

void InferenceServer::run() {
  net::io_context ioc(1);
  tcp::acceptor acceptor(ioc, tcp::endpoint(tcp::v4(), port_));
  std::cout << "[server] listening on :" << port_ << std::endl;

  while (running_.load(std::memory_order_relaxed)) {
    beast::error_code ec;
    tcp::socket socket(ioc);
    acceptor.accept(socket, ec);
    if (ec) {
      if (running_.load(std::memory_order_relaxed)) {
        std::cerr << "[server] accept error: " << ec.message() << std::endl;
      }
      continue;
    }

    net::post(workers_, [this, s = std::move(socket)]() mutable {
      handle_client(std::move(s));
    });
  }
}

void InferenceServer::handle_client(tcp::socket socket) {
  beast::error_code ec;
  beast::flat_buffer buffer;

  http::request<http::string_body> req;
  http::read(socket, buffer, req, ec);
  if (ec) {
    return;
  }

  http::response<http::string_body> res;
  res.version(req.version());
  res.set(http::field::server, "neuralcabin-cpp");
  res.set(http::field::content_type, "application/json");
  res.keep_alive(false);

  if (req.method() == http::verb::get && req.target() == "/healthz") {
    res.result(http::status::ok);
    res.body() = R"({"ok":true})";
  } else if (req.method() == http::verb::post && req.target() == "/infer") {
    try {
      if (store_.empty()) {
        res.result(http::status::service_unavailable);
        res.body() = R"({"error":"weights-not-loaded"})";
      } else {
        res.result(http::status::ok);
        res.body() = run_inference(req.body());
      }
    } catch (const std::exception& e) {
      res.result(http::status::internal_server_error);
      res.body() = std::string(R"({"error":")") + e.what() + R"("})";
    }
  } else {
    res.result(http::status::not_found);
    res.body() = R"({"error":"not-found"})";
  }

  res.prepare_payload();
  http::write(socket, res, ec);
  socket.shutdown(tcp::socket::shutdown_both, ec);
}

std::string InferenceServer::run_inference(const std::string& request_json) {
  const auto decrypted = store_.decrypt(weight_key_);
  std::ostringstream out;
  out << "{"
      << R"("ok":true,)"
      << R"("engine":"neuralcabin-cpp",)"
      << R"("requestBytes":)" << request_json.size() << ","
      << R"("loadedWeightBytes":)" << decrypted.size()
      << "}";
  return out.str();
}

} // namespace neuralcabin
