/*
 * http_server.h — Minimal single-threaded HTTP server for detection API.
 * No dependencies. Uses raw Winsock2/POSIX sockets.
 *
 * Endpoints:
 *   POST /detect   — upload image, get JSON detections
 *   GET  /health   — returns {"status":"ok"}
 *   GET  /stats    — returns inference stats
 */

#ifndef NN2_HTTP_SERVER_H
#define NN2_HTTP_SERVER_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSE_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
typedef int socket_t;
#define CLOSE_SOCKET close
#define INVALID_SOCKET -1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HTTP_MAX_BODY (10 * 1024 * 1024) /* 10MB max upload */

typedef struct {
    char method[8];
    char path[256];
    char content_type[128];
    int content_length;
    char* body;
} HttpRequest;

typedef void (*http_handler)(const HttpRequest* req, socket_t client, void* userdata);

static int http_parse_request(socket_t client, HttpRequest* req)
{
    char header[4096];
    int total = 0;
    memset(req, 0, sizeof(*req));
    req->body = NULL;

    /* Read header */
    while (total < (int)sizeof(header) - 1) {
        int n = recv(client, header + total, 1, 0);
        if (n <= 0) return -1;
        total++;
        if (total >= 4 && memcmp(header + total - 4, "\r\n\r\n", 4) == 0) break;
    }
    header[total] = 0;

    /* Parse first line */
    sscanf(header, "%7s %255s", req->method, req->path);

    /* Parse Content-Length */
    char* cl = strstr(header, "Content-Length:");
    if (!cl) cl = strstr(header, "content-length:");
    if (cl) req->content_length = atoi(cl + 15);

    /* Parse Content-Type */
    char* ct = strstr(header, "Content-Type:");
    if (!ct) ct = strstr(header, "content-type:");
    if (ct) {
        ct += 13;
        while (*ct == ' ') ct++;
        int i = 0;
        while (*ct && *ct != '\r' && *ct != '\n' && i < 127) req->content_type[i++] = *ct++;
        req->content_type[i] = 0;
    }

    /* Read body */
    if (req->content_length > 0 && req->content_length < HTTP_MAX_BODY) {
        req->body = (char*)malloc(req->content_length);
        int read = 0;
        while (read < req->content_length) {
            int n = recv(client, req->body + read, req->content_length - read, 0);
            if (n <= 0) break;
            read += n;
        }
    }
    return 0;
}

static void http_send(socket_t client, int code, const char* content_type, const char* body, int body_len)
{
    char header[512];
    const char* status = (code == 200) ? "OK" : (code == 404) ? "Not Found" : "Bad Request";
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, status, content_type, body_len);
    send(client, header, hlen, 0);
    if (body && body_len > 0)
        send(client, body, body_len, 0);
}

static void http_send_json(socket_t client, int code, const char* json)
{
    http_send(client, code, "application/json", json, (int)strlen(json));
}

static int http_listen(int port, http_handler handler, void* userdata)
{
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    socket_t server = socket(AF_INET, SOCK_STREAM, 0);
    if (server == INVALID_SOCKET) return -1;

    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        CLOSE_SOCKET(server);
        return -1;
    }

    listen(server, 8);
    fprintf(stderr, "nn2 HTTP server listening on port %d\n", port);

    while (1) {
        socket_t client = accept(server, NULL, NULL);
        if (client == INVALID_SOCKET) continue;

        HttpRequest req;
        if (http_parse_request(client, &req) == 0)
            handler(&req, client, userdata);

        free(req.body);
        CLOSE_SOCKET(client);
    }

    CLOSE_SOCKET(server);
    return 0;
}

#endif /* NN2_HTTP_SERVER_H */
