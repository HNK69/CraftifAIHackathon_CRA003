// event_sink.h -- where the overlay metadata goes.
//
// A tiny TCP server: it accepts consoles and writes one JSON line per event to
// each. No library, no broker, no dependency -- the DeepStream image ships
// libsrt's runtime but no headers, so a .so cannot link SRT, and a few hundred
// bytes a second of text gains nothing from SRT's ARQ anyway. The VIDEO is the
// loss-sensitive megabit stream and that goes over SRT (see run.sh).
//
// Non-blocking throughout, because this runs inside the pipeline's data path:
//
//   * accept() never waits -- a frame must not stall because nobody connected
//   * send() never waits -- a console that stops reading gets disconnected
//     rather than backing pressure up into the encoder
//
// A dropped console is the right failure. Metadata is a live view of the present;
// replaying a stalled consumer's backlog would show it boxes for frames that
// went past thirty seconds ago.
//
// It also does the TILE MAPPING. nvmultistreamtiler composes N sources onto one
// canvas, so geometry from source i has to be shifted into tile i before it
// describes anything the console can draw over the composed frame.

#ifndef EVENT_SINK_H_
#define EVENT_SINK_H_

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace event_sink {

inline const char* env_or(const char* k, const char* d) {
    const char* v = getenv(k);
    return (v && *v) ? v : d;
}

// Where source `i` sits on the tiler canvas, and how big that canvas is. run.sh
// exports these because it is what chose the grid.
struct Tiles {
    int tile_w = 1280, tile_h = 720, cols = 1;
    int space_w = 1280, space_h = 720;

    void load() {
        tile_w  = atoi(env_or("VMS_TILE_W", "1280"));
        tile_h  = atoi(env_or("VMS_TILE_H", "720"));
        cols    = atoi(env_or("VMS_TILE_COLS", "1"));
        space_w = atoi(env_or("VMS_SPACE_W", "1280"));
        space_h = atoi(env_or("VMS_SPACE_H", "720"));
        if (cols < 1) cols = 1;
    }
    double x_of(unsigned source_id) const { return double((source_id % (unsigned) cols) * tile_w); }
    double y_of(unsigned source_id) const { return double((source_id / (unsigned) cols) * tile_h); }
};

class Server {
public:
    ~Server() {
        for (size_t i = 0; i < m_clients.size(); ++i) close(m_clients[i]);
        if (m_listen >= 0) close(m_listen);
    }

    bool start(int port) {
        // A console that dies mid-write would otherwise deliver SIGPIPE and kill
        // the whole pipeline. MSG_NOSIGNAL covers send(), this covers the rest.
        signal(SIGPIPE, SIG_IGN);

        m_listen = socket(AF_INET, SOCK_STREAM, 0);
        if (m_listen < 0) return false;
        int one = 1;
        setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        fcntl(m_listen, F_SETFL, fcntl(m_listen, F_GETFL, 0) | O_NONBLOCK);

        struct sockaddr_in a;
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons((uint16_t) port);
        if (bind(m_listen, (struct sockaddr*) &a, sizeof(a)) != 0
            || listen(m_listen, 8) != 0) {
            fprintf(stderr, "[event-sink] cannot listen on :%d: %s\n",
                    port, strerror(errno));
            close(m_listen);
            m_listen = -1;
            return false;
        }
        m_port = port;
        fprintf(stderr, "[event-sink] metadata on tcp://:%d\n", port);
        return true;
    }

    // Take any console that has connected since the last call.
    void poll_accept() {
        if (m_listen < 0) return;
        for (;;) {
            const int fd = accept(m_listen, NULL, NULL);
            if (fd < 0) break;                       // EAGAIN: nobody waiting
            fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
            int one = 1;
            // Overlay data is latency-critical and tiny; Nagle would hold a line
            // back waiting for a bigger one that never comes.
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            m_clients.push_back(fd);
            fprintf(stderr, "[event-sink] console connected (%zu)\n", m_clients.size());
        }
    }

    void broadcast(const std::string& line) {
        for (size_t i = 0; i < m_clients.size();) {
            const ssize_t n = send(m_clients[i], line.data(), line.size(),
                                   MSG_NOSIGNAL | MSG_DONTWAIT);
            if (n == (ssize_t) line.size()) { ++i; continue; }
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                // The console is not draining. Drop it rather than block the
                // pipeline or queue a backlog it would render far too late.
                fprintf(stderr, "[event-sink] console not reading -- dropped\n");
            }
            close(m_clients[i]);
            m_clients.erase(m_clients.begin() + (long) i);
        }
    }

    size_t clients() const { return m_clients.size(); }
    int port() const { return m_port; }

private:
    int m_listen = -1;
    int m_port = 0;
    std::vector<int> m_clients;
};

}  // namespace event_sink

#endif  // EVENT_SINK_H_
