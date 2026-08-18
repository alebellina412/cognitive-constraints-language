/*
 * Lossless compressed-export wrapper for the archived UMT_dynamic.c.
 *
 * The model implementation below is included verbatim from ../legacy/.
 * Only fopen/fprintf/fclose for its full trajectory TSV are intercepted.
 * The generated .cltraj.zst file can reconstruct every original t, D, estr
 * row exactly; no simulation rule, parameter, RNG call, or update changes.
 */

#define _DEFAULT_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_CAPACITY 65536U

static FILE *trajectory_stream = NULL;
static uint32_t event_buffer[BLOCK_CAPACITY];
static uint8_t increment_bits[BLOCK_CAPACITY / 8U];
static uint32_t buffered_events = 0;
static int have_previous = 0;
static uint32_t previous_d = 0;
static uint32_t previous_estr = 0;

static uint64_t header_t = 0;
static uint32_t header_n0 = 0, header_d0 = 0;
static uint32_t header_seed = 1;
static double header_rho = 0.0, header_nu = 0.0, header_p = 0.0;

/* Defined before the macro substitutions below. */
static FILE *plain_fopen(const char *path, const char *mode) { return fopen(path, mode); }
static int plain_fclose(FILE *stream) { return fclose(stream); }

static void checked_write(const void *ptr, size_t size, size_t count, FILE *stream)
{
    if (fwrite(ptr, size, count, stream) != count) {
        perror("compressed trajectory write");
        exit(EXIT_FAILURE);
    }
}

static void flush_block(void)
{
    uint32_t count = buffered_events;
    uint32_t bit_bytes = (count + 7U) / 8U;
    if (count == 0) return;
    checked_write(&count, sizeof(count), 1, trajectory_stream);
    checked_write(event_buffer, sizeof(event_buffer[0]), count, trajectory_stream);
    checked_write(increment_bits, sizeof(increment_bits[0]), bit_bytes, trajectory_stream);
    buffered_events = 0;
    memset(increment_bits, 0, sizeof(increment_bits));
}

static void append_event(uint32_t estr, int increment_after)
{
    if (buffered_events == BLOCK_CAPACITY) flush_block();
    event_buffer[buffered_events] = estr;
    if (increment_after) increment_bits[buffered_events / 8U] |= (uint8_t)(1U << (buffered_events % 8U));
    buffered_events++;
}

static void write_header(FILE *stream)
{
    const char magic[8] = {'C', 'L', 'T', 'R', 'J', '1', '\0', '\0'};
    const uint32_t version = 1;
    const uint32_t block_capacity = BLOCK_CAPACITY;
    checked_write(magic, sizeof(magic), 1, stream);
    checked_write(&version, sizeof(version), 1, stream);
    checked_write(&header_t, sizeof(header_t), 1, stream);
    checked_write(&header_n0, sizeof(header_n0), 1, stream);
    checked_write(&header_d0, sizeof(header_d0), 1, stream);
    checked_write(&header_seed, sizeof(header_seed), 1, stream);
    checked_write(&block_capacity, sizeof(block_capacity), 1, stream);
    checked_write(&header_rho, sizeof(header_rho), 1, stream);
    checked_write(&header_nu, sizeof(header_nu), 1, stream);
    checked_write(&header_p, sizeof(header_p), 1, stream);
}

static int is_trajectory_path(const char *path)
{
    return strstr(path, "/UMT_dynamic_") != NULL && strstr(path, ".dat") != NULL;
}

static FILE *compressed_fopen(const char *path, const char *mode)
{
    char command[512], output[256];
    const char *requested_output;
    size_t length;
    if (!is_trajectory_path(path)) return plain_fopen(path, mode);
    requested_output = getenv("CLTRAJ_OUTPUT");
    if (requested_output != NULL && requested_output[0] != '\0') {
        snprintf(output, sizeof(output), "%s", requested_output);
    } else {
        snprintf(output, sizeof(output), "%s", path);
        length = strlen(output);
        if (length >= 4 && strcmp(output + length - 4, ".dat") == 0) output[length - 4] = '\0';
        strncat(output, ".cltraj.zst", sizeof(output) - strlen(output) - 1);
    }
    /* One compression thread and level 3 keep CPU usage deliberately bounded. */
    snprintf(command, sizeof(command), "zstd -q -T1 -3 -o '%s'", output);
    trajectory_stream = popen(command, "w");
    if (trajectory_stream == NULL) {
        perror("popen zstd");
        exit(EXIT_FAILURE);
    }
    write_header(trajectory_stream);
    return trajectory_stream;
}

static int compressed_fprintf(FILE *stream, const char *format, ...)
{
    va_list args;
    int result;
    if (stream != trajectory_stream) {
        va_start(args, format);
        result = vfprintf(stream, format, args);
        va_end(args);
        return result;
    }
    /* The archived program writes precisely "%d\\t%d\\t%d\\n" here. */
    va_start(args, format);
    (void)va_arg(args, int); /* t is implicit from record order. */
    {
        uint32_t d = (uint32_t)va_arg(args, int);
        uint32_t estr = (uint32_t)va_arg(args, int);
        if (have_previous) append_event(previous_estr, d > previous_d);
        previous_d = d;
        previous_estr = estr;
        have_previous = 1;
    }
    va_end(args);
    return 0;
}

static int compressed_fclose(FILE *stream)
{
    int result;
    if (stream != trajectory_stream) return plain_fclose(stream);
    if (have_previous) append_event(previous_estr, 0);
    flush_block();
    result = pclose(trajectory_stream);
    trajectory_stream = NULL;
    return result;
}

static void configure_header(int argc, char *argv[])
{
    if (argc == 7) {
        header_n0 = (uint32_t)atoi(argv[1]);
        header_t = (uint64_t)strtoull(argv[2], NULL, 10);
        header_rho = atof(argv[3]);
        header_nu = atof(argv[4]);
        header_p = atof(argv[5]);
        header_d0 = (uint32_t)atoi(argv[6]);
    }
}

#define main archived_model_main
#define fopen compressed_fopen
#define fprintf compressed_fprintf
#define fclose compressed_fclose
#include "../legacy/UMT_dynamic.c"
#undef main
#undef fopen
#undef fprintf
#undef fclose

int main(int argc, char *argv[])
{
    configure_header(argc, argv);
    return archived_model_main(argc, argv);
}
