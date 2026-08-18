/*
 * Fast time-dependent UMT simulator.
 *
 * This keeps the stochastic rules and command-line interface of
 * legacy/UMT_dynamic.c, but represents the urn by a Fenwick tree and the
 * stream by the realised history.  Thus urn draws and updates cost O(log D),
 * while stream draws cost O(1).  The full trajectory is written directly in
 * the existing CLTRJ1 compressed format.
 */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define NARGC 7
#define BLOCK_CAPACITY 65536U

typedef struct {
    FILE *stream;
    uint32_t events[BLOCK_CAPACITY];
    uint8_t increments[BLOCK_CAPACITY / 8U];
    uint32_t count;
} trajectory_writer;

static void checked_write(const void *ptr, size_t size, size_t count, FILE *stream)
{
    if (fwrite(ptr, size, count, stream) != count) {
        perror("compressed trajectory write");
        exit(EXIT_FAILURE);
    }
}

static void flush_block(trajectory_writer *writer)
{
    uint32_t bit_bytes;
    if (writer->count == 0) return;
    bit_bytes = (writer->count + 7U) / 8U;
    checked_write(&writer->count, sizeof(writer->count), 1, writer->stream);
    checked_write(writer->events, sizeof(writer->events[0]), writer->count, writer->stream);
    checked_write(writer->increments, sizeof(writer->increments[0]), bit_bytes, writer->stream);
    writer->count = 0;
    memset(writer->increments, 0, sizeof(writer->increments));
}

static void append_event(trajectory_writer *writer, uint32_t event, int increment_after)
{
    if (writer->count == BLOCK_CAPACITY) flush_block(writer);
    writer->events[writer->count] = event;
    if (increment_after)
        writer->increments[writer->count / 8U] |= (uint8_t)(1U << (writer->count % 8U));
    writer->count++;
}

static void write_header(FILE *stream, uint64_t steps, uint32_t n0, uint32_t d0,
                         double rho, double nu, double p)
{
    const char magic[8] = {'C', 'L', 'T', 'R', 'J', '1', '\0', '\0'};
    const uint32_t version = 1, seed = 1, block_capacity = BLOCK_CAPACITY;
    checked_write(magic, sizeof(magic), 1, stream);
    checked_write(&version, sizeof(version), 1, stream);
    checked_write(&steps, sizeof(steps), 1, stream);
    checked_write(&n0, sizeof(n0), 1, stream);
    checked_write(&d0, sizeof(d0), 1, stream);
    checked_write(&seed, sizeof(seed), 1, stream);
    checked_write(&block_capacity, sizeof(block_capacity), 1, stream);
    checked_write(&rho, sizeof(rho), 1, stream);
    checked_write(&nu, sizeof(nu), 1, stream);
    checked_write(&p, sizeof(p), 1, stream);
}

static FILE *open_trajectory(const char *default_path, uint64_t steps, uint32_t n0, uint32_t d0,
                             double rho, double nu, double p)
{
    char command[1024];
    const char *path = getenv("CLTRAJ_OUTPUT");
    if (path == NULL || path[0] == '\0') path = default_path;
    if (snprintf(command, sizeof(command), "zstd -q -T1 -3 -o '%s'", path) >= (int)sizeof(command)) {
        fprintf(stderr, "Compressed trajectory path is too long.\n");
        return NULL;
    }
    FILE *stream = popen(command, "w");
    if (stream == NULL) return NULL;
    write_header(stream, steps, n0, d0, rho, nu, p);
    return stream;
}

static void fenwick_add(double *tree, int n, int index, double delta)
{
    while (index <= n) {
        tree[index] += delta;
        index += index & -index;
    }
}

static double fenwick_prefix_sum(const double *tree, int index)
{
    double total = 0.0;
    while (index > 0) {
        total += tree[index];
        index -= index & -index;
    }
    return total;
}

static double fenwick_value(const double *tree, int index)
{
    return fenwick_prefix_sum(tree, index) - fenwick_prefix_sum(tree, index - 1);
}

/* Return the first positive-mass index whose prefix sum is greater than draw. */
static int fenwick_find_draw(const double *tree, int n, double draw)
{
    int index = 0, bit = 1;
    while ((bit << 1) <= n) bit <<= 1;
    while (bit > 0) {
        int next = index + bit;
        if (next <= n && tree[next] <= draw) {
            index = next;
            draw -= tree[next];
        }
        bit >>= 1;
    }
    return index + 1;
}

int main(int argc, char *argv[])
{
    int N0, T, D0, D = 0, n_objects = 1;
    double rho, nu, p, alpha, total_mass;
    char trajectory_path[256], frequency_path[256];
    FILE *frequency_file;
    trajectory_writer writer = {0};

    if (argc != NARGC) {
        fprintf(stderr, "Usage: %s <N0> <T> <rho> <nu> <p> <D0>\n", argv[0]);
        return EXIT_FAILURE;
    }
    N0 = atoi(argv[1]);
    T = atoi(argv[2]);
    rho = atof(argv[3]);
    nu = atof(argv[4]);
    p = atof(argv[5]);
    D0 = atoi(argv[6]);
    if (N0 <= 0 || T <= 0 || D0 <= 0) {
        fprintf(stderr, "N0, T and D0 must be positive.\n");
        return EXIT_FAILURE;
    }
    if (rho < 0.0 || nu < 0.0 || p < 0.0 || p > 1.0) {
        fprintf(stderr, "Require rho, nu >= 0 and 0 <= p <= 1.\n");
        return EXIT_FAILURE;
    }
    alpha = 1.0 / (double)D0;

    /* At most T discoveries plus the active frontier. */
    int capacity = T + 1;
    uint32_t *history = malloc((size_t)T * sizeof(*history));
    uint32_t *frequency = calloc((size_t)capacity, sizeof(*frequency));
    int *first_time = calloc((size_t)capacity, sizeof(*first_time));
    double *urn = calloc((size_t)capacity + 1U, sizeof(*urn));
    if (history == NULL || frequency == NULL || first_time == NULL || urn == NULL) {
        perror("allocating fast simulator state");
        free(history); free(frequency); free(first_time); free(urn);
        return EXIT_FAILURE;
    }

    /* This is exactly the legacy initial urn: one frontier item of mass N0. */
    total_mass = (double)N0;
    fenwick_add(urn, capacity, 1, total_mass);

    if (mkdir("data", 0777) != 0 && errno != EEXIST) {
        perror("mkdir data");
        free(history); free(frequency); free(first_time); free(urn);
        return EXIT_FAILURE;
    }
    snprintf(trajectory_path, sizeof(trajectory_path),
             "data/UMT_dynamic_fast_rho=%.1f_nu=%.1f_p=%.2f_N0=%d_D0=%d.cltraj.zst",
             rho, nu, p, N0, D0);
    snprintf(frequency_path, sizeof(frequency_path),
             "data/n_UMT_dynamic_fast_rho=%.1f_nu=%.1f_p=%.2f_N0=%d_D0=%d.dat",
             rho, nu, p, N0, D0);
    writer.stream = open_trajectory(trajectory_path, (uint64_t)T, (uint32_t)N0, (uint32_t)D0,
                                    rho, nu, p);
    if (writer.stream == NULL) {
        perror("opening zstd trajectory");
        free(history); free(frequency); free(first_time); free(urn);
        return EXIT_FAILURE;
    }
    frequency_file = fopen(frequency_path, "w");
    if (frequency_file == NULL) {
        perror("opening final frequency output");
        pclose(writer.stream);
        free(history); free(frequency); free(first_time); free(urn);
        return EXIT_FAILURE;
    }

    srand48(1);
    for (int t = 0; t < T; t++) {
        double ran = lrand48() / (RAND_MAX + 1.0);
        double scaled_d = alpha * (double)D;
        double lazy_probability = p * (scaled_d * scaled_d / (scaled_d * scaled_d + 1.0));
        int selected;

        if (ran < 1.0 - lazy_probability) {
            double draw;
            ran = lrand48() / (RAND_MAX + 1.0);
            draw = ran * total_mass;
            if (draw >= total_mass) draw = nextafter(total_mass, 0.0);
            selected = fenwick_find_draw(urn, capacity, draw) - 1;
        } else {
            /* The legacy stream distribution is exactly uniform over history[0..t). */
            ran = lrand48() / (RAND_MAX + 1.0);
            int history_index = (int)(ran * (double)t);
            if (history_index >= t) history_index = t - 1;
            selected = (int)history[history_index];
        }

        if (selected < 0 || selected >= n_objects) {
            fprintf(stderr, "Internal urn-state error at t=%d.\n", t);
            fclose(frequency_file);
            pclose(writer.stream);
            free(history); free(frequency); free(first_time); free(urn);
            return EXIT_FAILURE;
        }

        history[t] = (uint32_t)selected;
        frequency[selected] += 1U;
        if (selected == n_objects - 1) {
            double prior_frontier_mass = fenwick_value(urn, selected + 1);
            first_time[selected] = t;
            D++;
            fenwick_add(urn, capacity, selected + 1, rho - prior_frontier_mass);
            fenwick_add(urn, capacity, selected + 2, prior_frontier_mass + nu);
            total_mass += rho + nu;
            n_objects++;
            append_event(&writer, (uint32_t)selected, 1);
        } else {
            fenwick_add(urn, capacity, selected + 1, rho);
            total_mass += rho;
            append_event(&writer, (uint32_t)selected, 0);
        }
    }

    flush_block(&writer);
    if (pclose(writer.stream) != 0) {
        fprintf(stderr, "zstd failed while writing the trajectory.\n");
        fclose(frequency_file);
        free(history); free(frequency); free(first_time); free(urn);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < n_objects; i++) {
        double weight = fenwick_value(urn, i + 1);
        fprintf(frequency_file, "%d\t%u\t%.17g\t%.17g\n",
                first_time[i], frequency[i], weight, weight / total_mass);
    }
    fclose(frequency_file);
    free(history); free(frequency); free(first_time); free(urn);
    return EXIT_SUCCESS;
}
