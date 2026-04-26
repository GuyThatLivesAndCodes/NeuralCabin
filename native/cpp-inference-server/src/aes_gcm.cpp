#include "neuralcabin/aes_gcm.hpp"

#include <stdexcept>

#include <openssl/evp.h>
#include <openssl/rand.h>

namespace neuralcabin {

namespace {

[[noreturn]] void throw_crypto(const char* msg) {
  throw std::runtime_error(msg);
}

} // namespace

Aes256GcmPayload encrypt_aes256_gcm(
  std::span<const std::uint8_t> plaintext,
  std::span<const std::uint8_t, 32> key
) {
  Aes256GcmPayload payload;
  payload.ciphertext.resize(plaintext.size());

  if (RAND_bytes(payload.nonce.data(), static_cast<int>(payload.nonce.size())) != 1) {
    throw_crypto("RAND_bytes failed");
  }

  EVP_CIPHER_CTX* raw_ctx = EVP_CIPHER_CTX_new();
  if (!raw_ctx) throw_crypto("EVP_CIPHER_CTX_new failed");

  int len = 0;
  int out_len = 0;

  if (EVP_EncryptInit_ex(raw_ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_EncryptInit_ex failed");
  }

  if (EVP_CIPHER_CTX_ctrl(raw_ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(payload.nonce.size()), nullptr) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_CIPHER_CTX_ctrl set ivlen failed");
  }

  if (EVP_EncryptInit_ex(raw_ctx, nullptr, nullptr, key.data(), payload.nonce.data()) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_EncryptInit_ex set key/iv failed");
  }

  if (EVP_EncryptUpdate(
        raw_ctx,
        payload.ciphertext.data(),
        &len,
        plaintext.data(),
        static_cast<int>(plaintext.size())
      ) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_EncryptUpdate failed");
  }
  out_len += len;

  if (EVP_EncryptFinal_ex(raw_ctx, payload.ciphertext.data() + out_len, &len) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_EncryptFinal_ex failed");
  }
  out_len += len;
  payload.ciphertext.resize(static_cast<std::size_t>(out_len));

  if (EVP_CIPHER_CTX_ctrl(raw_ctx, EVP_CTRL_GCM_GET_TAG, static_cast<int>(payload.tag.size()), payload.tag.data()) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_CIPHER_CTX_ctrl get tag failed");
  }

  EVP_CIPHER_CTX_free(raw_ctx);
  return payload;
}

std::vector<std::uint8_t> decrypt_aes256_gcm(
  const Aes256GcmPayload& payload,
  std::span<const std::uint8_t, 32> key
) {
  std::vector<std::uint8_t> plain(payload.ciphertext.size());

  EVP_CIPHER_CTX* raw_ctx = EVP_CIPHER_CTX_new();
  if (!raw_ctx) throw_crypto("EVP_CIPHER_CTX_new failed");

  int len = 0;
  int out_len = 0;

  if (EVP_DecryptInit_ex(raw_ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_DecryptInit_ex failed");
  }

  if (EVP_CIPHER_CTX_ctrl(raw_ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(payload.nonce.size()), nullptr) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_CIPHER_CTX_ctrl set ivlen failed");
  }

  if (EVP_DecryptInit_ex(raw_ctx, nullptr, nullptr, key.data(), payload.nonce.data()) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_DecryptInit_ex set key/iv failed");
  }

  if (EVP_DecryptUpdate(
        raw_ctx,
        plain.data(),
        &len,
        payload.ciphertext.data(),
        static_cast<int>(payload.ciphertext.size())
      ) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_DecryptUpdate failed");
  }
  out_len += len;

  if (EVP_CIPHER_CTX_ctrl(
        raw_ctx,
        EVP_CTRL_GCM_SET_TAG,
        static_cast<int>(payload.tag.size()),
        const_cast<std::uint8_t*>(payload.tag.data())
      ) != 1) {
    EVP_CIPHER_CTX_free(raw_ctx);
    throw_crypto("EVP_CIPHER_CTX_ctrl set tag failed");
  }

  const int ok = EVP_DecryptFinal_ex(raw_ctx, plain.data() + out_len, &len);
  EVP_CIPHER_CTX_free(raw_ctx);
  if (ok != 1) throw_crypto("AES-256-GCM authentication failed");

  out_len += len;
  plain.resize(static_cast<std::size_t>(out_len));
  return plain;
}

} // namespace neuralcabin
