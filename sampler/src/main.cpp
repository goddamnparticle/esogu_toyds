#include "sampler.h"
#include <iostream>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

int counter;
int GNFiles;
std::vector<std::array<int, 5>> diagReport;

void printUsage(void);
void printHelp(void);

int main(int argc, char *argv[])
{
    // -----------> Default values <---------------- //
    std::string dataRoot("/home/yorek/sample/data/ESOGU_ToyDS_A/");
    std::string flags("pdcrh");
    int numPoints(1024);
    int ChunkSize(10);
    int recChunkSize(16);
    int c {};

    // -----------> Argument parsing <---------------- //
    while ((c = getopt(argc, argv, "p:d:c:r:h")) != -1) {
        switch (c) {
        case 'p':
            numPoints = atoi(optarg);
            break;
        case 'd':
            dataRoot = optarg;
            break;
        case 'c':
            ChunkSize = atoi(optarg);
            break;
        case 'r':
            recChunkSize = atoi(optarg);
            break;
        case 'h':
            printHelp();
            return 0;
        case '?':
            printUsage();
            return 1;
        default:
            printf("getopt func. returned unknown character.");
            exit(2);
        }
    }
    // Clear terminal screen
    printf("\033c");
    if (optind < argc) {
        std::string extraargs("Extra arguments not parsed : ");
        for (; optind < argc; optind++) {
            printf("\033c");
            extraargs = extraargs + argv[optind] + " ";
        }
        std::cout << extraargs;
    }

    MeshSampler sampler(numPoints, dataRoot, ChunkSize, recChunkSize);
    sampler.sampleData();
    printf("\33[2K\r"); // Delete and move cursor to the start of line
    std::cout << "Sampled               : " << counter << " / " << GNFiles << "\n";
    std::cout << "----------------------> Completed  <----------------------\n";

    return 0;
}
