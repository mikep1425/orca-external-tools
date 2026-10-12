/*
 * oet_client -- minimal compiled client for oet_server.
 *
 * ORCA starts an external tool once per gradient, so the client's start-up cost
 * is paid tens of thousands of times in a GOAT run. A Python client pays ~20 ms
 * of interpreter start before it does anything (plus ~70 ms more if it imports
 * requests); this binary does the same job in ~2-3 ms.
 *
 * It speaks the same HTTP/1.1 POST /calculate as oet_client / oet_client_fast,
 * so the server needs no changes. The server writes the result files itself; all
 * this does is forward the arguments and the working directory, relay the
 * server's stdout and propagate the exit status.
 *
 * Named oet_client_c, not oet_client: the package already installs a console
 * script called oet_client (the requests-based Python client) into the venv's
 * bin, which would shadow a same-named binary on PATH.
 *
 * Build:
 *   cc -O2 -std=c11 -o oet_client_c oet_client.c        (make)
 *   cc -O2 -std=c11 -static -o oet_client_c oet_client.c (make static)
 *
 * Of the ~2 ms a call takes, ~1.3 ms is fork+exec of the process itself, which
 * ORCA's external-tool interface makes unavoidable. A numeric host (127.0.0.1
 * rather than localhost) skips getaddrinfo and the NSS machinery behind it;
 * a static build saves a further ~0.3 ms of dynamic linking, and is safe as
 * long as only numeric hosts are used -- statically linked getaddrinfo needs
 * the glibc NSS shared objects at runtime anyway.
 *
 * Use (from an ORCA input):
 *   progext "/path/to/oet_client_c"
 *   ext_params "--method gfn2"
 */

/* POSIX networking + getcwd under a strict -std=c11 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#define DEFAULT_HOST "127.0.0.1"
#define DEFAULT_PORT "8888"
#define READ_CHUNK 65536

/* ---------------------------------------------------------------- buffers */

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} buf_t;

static void buf_reserve(buf_t *b, size_t extra)
{
    if (b->len + extra + 1 <= b->cap)
        return;
    size_t cap = b->cap ? b->cap : 1024;
    while (cap < b->len + extra + 1)
        cap *= 2;
    char *p = realloc(b->data, cap);
    if (!p) {
        fprintf(stderr, "oet_client: out of memory\n");
        exit(1);
    }
    b->data = p;
    b->cap  = cap;
}

static void buf_append(buf_t *b, const char *s, size_t n)
{
    buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void buf_puts(buf_t *b, const char *s) { buf_append(b, s, strlen(s)); }

/* ------------------------------------------------------------------- json */

/* Append s as a quoted JSON string. Only the escapes JSON requires. */
static void json_quote(buf_t *b, const char *s)
{
    buf_puts(b, "\"");
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n");  break;
        case '\r': buf_puts(b, "\\r");  break;
        case '\t': buf_puts(b, "\\t");  break;
        default:
            if (*p < 0x20) {
                char esc[8];
                snprintf(esc, sizeof esc, "\\u%04x", *p);
                buf_puts(b, esc);
            } else {
                buf_append(b, (const char *)p, 1);
            }
        }
    }
    buf_puts(b, "\"");
}

/*
 * Locate the value of a top-level string field. Returns a pointer to the first
 * character inside the quotes and sets *len to the raw (still escaped) length,
 * or NULL if the field is absent or not a string.
 */
static const char *json_find_string(const char *body, const char *field, size_t *len)
{
    char pattern[64];
    snprintf(pattern, sizeof pattern, "\"%s\"", field);
    const char *p = strstr(body, pattern);
    if (!p)
        return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t')
        p++;
    if (*p++ != ':')
        return NULL;
    while (*p == ' ' || *p == '\t')
        p++;
    if (*p++ != '"')
        return NULL;

    const char *start = p;
    while (*p) {
        if (*p == '\\' && p[1]) {
            p += 2;
            continue;
        }
        if (*p == '"')
            break;
        p++;
    }
    if (*p != '"')
        return NULL;
    *len = (size_t)(p - start);
    return start;
}

/* Write an escaped JSON string body to stream, undoing the escapes. */
static void json_unescape_to(FILE *stream, const char *s, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        if (s[i] != '\\' || i + 1 >= len) {
            fputc(s[i], stream);
            continue;
        }
        switch (s[++i]) {
        case 'n':  fputc('\n', stream); break;
        case 'r':  fputc('\r', stream); break;
        case 't':  fputc('\t', stream); break;
        case 'b':  fputc('\b', stream); break;
        case 'f':  fputc('\f', stream); break;
        case '"':  fputc('"',  stream); break;
        case '\\': fputc('\\', stream); break;
        case '/':  fputc('/',  stream); break;
        case 'u': {
            /* Only the BMP range that matters for program output. */
            if (i + 4 < len) {
                char hex[5] = { s[i + 1], s[i + 2], s[i + 3], s[i + 4], 0 };
                long cp = strtol(hex, NULL, 16);
                i += 4;
                if (cp < 0x80) {
                    fputc((int)cp, stream);
                } else if (cp < 0x800) {
                    fputc((int)(0xC0 | (cp >> 6)), stream);
                    fputc((int)(0x80 | (cp & 0x3F)), stream);
                } else {
                    fputc((int)(0xE0 | (cp >> 12)), stream);
                    fputc((int)(0x80 | ((cp >> 6) & 0x3F)), stream);
                    fputc((int)(0x80 | (cp & 0x3F)), stream);
                }
            }
            break;
        }
        default:
            fputc(s[i], stream);
        }
    }
}

/* ------------------------------------------------------------------- http */

/* Decode a chunked body in place. Returns the new length. */
static size_t dechunk(char *body, size_t len)
{
    size_t out = 0, pos = 0;
    while (pos < len) {
        char *eol = memchr(body + pos, '\r', len - pos);
        if (!eol)
            break;
        size_t size = (size_t)strtoul(body + pos, NULL, 16);
        pos = (size_t)(eol - body) + 2;
        if (size == 0 || pos + size > len)
            break;
        memmove(body + out, body + pos, size);
        out += size;
        pos += size + 2;
    }
    body[out] = '\0';
    return out;
}

/*
 * Connect to a numeric host without going through getaddrinfo, which pulls in
 * the NSS machinery (and on a cluster can mean LDAP lookups). Returns -2 when
 * the host is not a numeric address, so the caller can fall back.
 */
static int connect_numeric(const char *host, const char *port)
{
    struct sockaddr_in  a4;
    struct sockaddr_in6 a6;
    struct sockaddr *sa;
    socklen_t salen;
    int family;

    char *end;
    long p = strtol(port, &end, 10);
    if (*end != '\0' || p <= 0 || p > 65535)
        return -2;

    memset(&a4, 0, sizeof a4);
    memset(&a6, 0, sizeof a6);
    if (inet_pton(AF_INET, host, &a4.sin_addr) == 1) {
        a4.sin_family = AF_INET;
        a4.sin_port   = htons((uint16_t)p);
        sa = (struct sockaddr *)&a4;
        salen = sizeof a4;
        family = AF_INET;
    } else if (inet_pton(AF_INET6, host, &a6.sin6_addr) == 1) {
        a6.sin6_family = AF_INET6;
        a6.sin6_port   = htons((uint16_t)p);
        sa = (struct sockaddr *)&a6;
        salen = sizeof a6;
        family = AF_INET6;
    } else {
        return -2;
    }

    int fd = socket(family, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    if (connect(fd, sa, salen) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int connect_to(const char *host, const char *port)
{
    int fd = connect_numeric(host, port);
    if (fd == -1) {
        fprintf(stderr, "oet_client: cannot connect to %s:%s: %s\n",
                host, port, strerror(errno));
        return -1;
    }
    if (fd == -2) {
        /* Not a numeric address, so resolve it. */
        struct addrinfo hints, *res, *rp;
        memset(&hints, 0, sizeof hints);
        hints.ai_family   = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        /* Skip address families the host has no address for, and skip the
         * /etc/services lookup for what is always a numeric port. */
        hints.ai_flags    = AI_ADDRCONFIG | AI_NUMERICSERV;

        int rc = getaddrinfo(host, port, &hints, &res);
        if (rc != 0) {
            fprintf(stderr, "oet_client: cannot resolve %s:%s: %s\n",
                    host, port, gai_strerror(rc));
            return -1;
        }
        for (rp = res; rp; rp = rp->ai_next) {
            fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
            if (fd < 0)
                continue;
            if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0)
                break;
            close(fd);
            fd = -1;
        }
        freeaddrinfo(res);
        if (fd < 0) {
            fprintf(stderr, "oet_client: cannot connect to %s:%s: %s\n",
                    host, port, strerror(errno));
            return -1;
        }
    }

    /* The request is one small write and the reply is wanted immediately, so
     * do not let Nagle hold a segment waiting for more data. */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

static int send_all(int fd, const char *data, size_t len)
{
    while (len > 0) {
        ssize_t n = write(fd, data, len);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
        data += n;
        len  -= (size_t)n;
    }
    return 0;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    char host[256] = DEFAULT_HOST;
    char port[32]  = DEFAULT_PORT;

    /* Build the JSON payload while scanning the arguments. */
    buf_t json = { 0 };
    buf_puts(&json, "{\"arguments\":[");

    int nforwarded = 0;
    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-b") || !strcmp(argv[i], "--bind")) && i + 1 < argc) {
            const char *bind = argv[++i];
            const char *colon = strrchr(bind, ':');
            if (colon) {
                size_t hlen = (size_t)(colon - bind);
                if (hlen >= sizeof host)
                    hlen = sizeof host - 1;
                memcpy(host, bind, hlen);
                host[hlen] = '\0';
                snprintf(port, sizeof port, "%s", colon + 1);
            } else {
                snprintf(host, sizeof host, "%s", bind);
            }
            continue;
        }
        if (nforwarded++)
            buf_puts(&json, ",");
        json_quote(&json, argv[i]);
    }

    char cwd[4096];
    if (!getcwd(cwd, sizeof cwd)) {
        fprintf(stderr, "oet_client: getcwd failed: %s\n", strerror(errno));
        return 1;
    }
    buf_puts(&json, "],\"directory\":");
    json_quote(&json, cwd);
    buf_puts(&json, "}");

    buf_t req = { 0 };
    char header[512];
    snprintf(header, sizeof header,
             "POST /calculate HTTP/1.1\r\n"
             "Host: %s:%s\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n",
             host, port, json.len);
    buf_puts(&req, header);
    buf_append(&req, json.data, json.len);

    int fd = connect_to(host, port);
    if (fd < 0)
        return 1;
    if (send_all(fd, req.data, req.len) != 0) {
        fprintf(stderr, "oet_client: send failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    buf_t resp = { 0 };
    for (;;) {
        buf_reserve(&resp, READ_CHUNK);
        ssize_t n = read(fd, resp.data + resp.len, READ_CHUNK);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "oet_client: read failed: %s\n", strerror(errno));
            close(fd);
            return 1;
        }
        if (n == 0)
            break;
        resp.len += (size_t)n;
        resp.data[resp.len] = '\0';
    }
    close(fd);

    char *sep = resp.data ? strstr(resp.data, "\r\n\r\n") : NULL;
    if (!sep) {
        fprintf(stderr, "oet_client: malformed response from server\n");
        return 1;
    }
    *sep = '\0';
    char *body = sep + 4;
    size_t body_len = resp.len - (size_t)(body - resp.data);

    /* waitress normally sends Content-Length, but handle chunked too. */
    for (char *p = resp.data; *p; p++)
        *p = (char)tolower((unsigned char)*p);
    if (strstr(resp.data, "transfer-encoding: chunked"))
        body_len = dechunk(body, body_len);
    (void)body_len;

    size_t len = 0;
    const char *out = json_find_string(body, "stdout", &len);
    if (out)
        json_unescape_to(stdout, out, len);
    fflush(stdout);

    const char *status = json_find_string(body, "status", &len);
    if (!status) {
        fprintf(stderr, "oet_client: no status in response: %s\n", body);
        return 1;
    }
    if (len == 7 && !strncmp(status, "Success", 7))
        return 0;

    /* Mirror the Python client: report what the server said, then fail. */
    const char *etype = json_find_string(body, "error_type", &len);
    if (etype) {
        fprintf(stderr, "Server error ");
        json_unescape_to(stderr, etype, len);
        fprintf(stderr, ": ");
    }
    const char *emsg = json_find_string(body, "error_message", &len);
    if (emsg)
        json_unescape_to(stderr, emsg, len);
    fprintf(stderr, "\n");
    const char *tb = json_find_string(body, "traceback", &len);
    if (tb) {
        fprintf(stderr, "Exact traceback: ");
        json_unescape_to(stderr, tb, len);
        fprintf(stderr, "\n");
    }
    return 1;
}
