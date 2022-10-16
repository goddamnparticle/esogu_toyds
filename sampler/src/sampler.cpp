#include "sampler.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <sys/stat.h>

namespace fs = std::filesystem;
std::mutex gLock;

// // Random value generator
template <typename T>
T random(T range_from, T range_to)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);
    return distr(generator);
}
// Weighted choices `weights as std::vector`
int MeshSampler::weightedRandom(std::vector<double> weights) const
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return dist(generator);
}
// Get filepatsh in data directory.
int MeshSampler::findFilePaths(void)
{
    if (!fs::is_directory(fRoot))
        throw std::runtime_error(fRoot + " no such folder.");

    for (const auto &entry : fs::recursive_directory_iterator(fRoot)) {
        if (fs::is_regular_file(entry))
            fFiles.emplace_back(entry.path().string());
    }
    return fFiles.size();
}
// Read file
MeshSampler::VF MeshSampler::readFile(const std::string &path) const
{
    // Checks whether a file exist.
    struct stat buffer;
    assert(stat(path.c_str(), &buffer) == 0);
    // Load file.
    std::ifstream file(path);
    std::string line;

    int NumVertices {};
    int NumFaces {};

    // Pull number of verticies and faces
    while (line != "end_header" && getline(file, line)) {

        if (line.find("element vertex") != std::string::npos) {
            NumVertices = stoi(line.substr(line.rfind(' ')));
        }
        if (line.find("element face") != std::string::npos) {
            NumFaces = stoi(line.substr(line.rfind(' ')));
        }
    }

    double coord {};
    int vertpos {};
    MeshSampler::VF vf {};

    vf.Verticies.reserve(NumVertices);
    vf.Faces.reserve(NumFaces);

    for (int i = 0; i < NumVertices; i++) {
        getline(file, line);
        std::stringstream ss(line);
        vf.Verticies.push_back(std::array<double, 3> {});

        for (int j = 0; j < 3; j++) {
            ss >> coord;
            vf.Verticies[i][j] = coord;
        }
    }

    for (int i = 0; i < NumFaces; i++) {
        getline(file, line);
        std::stringstream ss(line);
        ss >> vertpos;
        vf.Faces.push_back(std::array<int, 3> {});
        for (int j = 0; j < 3; j++) {
            ss >> vertpos;
            vf.Faces[i][j] = vertpos;
        }
    }
    file.close();
    return vf;
}
// Calculate TriangleArea
double MeshSampler::TriangleArea(const Point &p1, const Point &p2, const Point &p3) const
{
    double u = 0.0;
    auto vecsubs = [](const auto &p1, const auto &p2) {
        std::vector<double> result;
        for (int i = 0; i < (int)p1.size(); i++)
            result.push_back(p1[i] - p2[i]);
        return result;
    };

    auto norm = [](std::vector<double> &p) {
        double sum(0.0);
        for (auto &coord : p)
            sum += coord * coord;
        return std::sqrt(sum);
    };

    auto p1p2 = vecsubs(p1, p2);
    auto p2p3 = vecsubs(p2, p3);
    auto p3p1 = vecsubs(p3, p1);
    auto side_a = norm(p1p2);
    auto side_b = norm(p2p3);
    auto side_c = norm(p3p1);
    u = 0.5 * (side_a + side_b + side_c);

    return std::sqrt(std::max(u * (u - side_a) * (u - side_b) * (u - side_c), 0.0));
}

Point MeshSampler::pointInTriangle(const Point &p1, const Point &p2, const Point &p3) const
{
    std::vector<double> s_t(2);
    std::array<double, 3> tempPoint;
    std::generate(s_t.begin(), s_t.end(), [&]() { return random(0.0, 1.0); });
    std::sort(s_t.begin(), s_t.end());

    for (int i = 0; i < 3; i++)
        tempPoint[i] = s_t[0] * p1[i] + (s_t[1] - s_t[0]) * p2[i] + (1.0 - s_t[1]) * p3[i];

    return tempPoint;
}

pSPoints MeshSampler::sampleOne(const VF &vf) const
{
    std::vector<double> Areas(vf.Faces.size());
    std::vector<Face> sampledFaces(fNPoints);
    pSPoints sampledPoints = std::make_unique<double[]>(fNPoints * 3);

    // Caluculate Triangles area
    for (std::size_t i = 0; i < Areas.size(); i++)
        Areas[i] = MeshSampler::TriangleArea(
            vf.Verticies[vf.Faces[i][0]],
            vf.Verticies[vf.Faces[i][1]],
            vf.Verticies[vf.Faces[i][2]]);

    if (vf.Faces.size() > 200000) {
        // const int ChunkSize = 64;
        const int TSize = fNPoints;
        const int NChunks = TSize / recChunkSize;
        const int Reminder = TSize % recChunkSize;
        std::vector<std::future<void>> threadPool;

        auto sample_iter = [&](const int &start, const int &end) {
            for (int i = start; i < end; i++) {
                auto rd = weightedRandom(Areas);
                sampledFaces[i] = vf.Faces[rd];
            }
        };
        for (int i = 0; i < NChunks; i++)
            threadPool.push_back(std::async(
                std::launch::async,
                sample_iter,
                i * ChunkSize,
                (i + 1) * ChunkSize));

        if (Reminder > 0)
            threadPool.push_back(std::async(
                std::launch::async,
                sample_iter,
                TSize - Reminder,
                TSize));

        for (int i = 0; i < (int)threadPool.size(); i++)
            threadPool[i].get();
    } else {
        for (int i = 0; i < fNPoints; i++)
            sampledFaces[i] = vf.Faces[weightedRandom(Areas)];
    }

    for (int i = 0; i < fNPoints; i++) {
        auto pointInT = pointInTriangle(
            vf.Verticies[sampledFaces[i][0]],
            vf.Verticies[sampledFaces[i][1]],
            vf.Verticies[sampledFaces[i][2]]);
        memmove(sampledPoints.get() + (i * 3), pointInT.data(), 3 * sizeof(double));
    }
    return sampledPoints;
}

void MeshSampler::Save(const std::string &path, const pSPoints &sampledPoints)
{
    std::string temp(path);
    std::string newpath = temp.replace(temp.rfind('A'), 1, "S" + std::to_string(fNPoints) + "AB");
    std::string folder = temp.replace(temp.rfind('/') + 1, std::string::npos, "");

    if (!fs::is_directory(folder))
        fs::create_directories(folder);

    std::ofstream of(newpath.replace(newpath.find('.'), std::string::npos, ".bin"), std::ios::binary);
    of.write(reinterpret_cast<const char *>(sampledPoints.get()), fNPoints * 3 * sizeof(double));
    of.close();

    gLock.lock();
    counter += 1;
    gLock.unlock();
}

void MeshSampler::sampleData(void)
{

    GNFiles = findFilePaths();
    const int NFiles = GNFiles;
    const int NChunks = NFiles / ChunkSize;
    const int Reminder = NFiles % ChunkSize;
    std::vector<std::future<void>> threadPool;
    int i;

    auto sample_iter = [&](const int &start, const int &end) {
        for (int i = start; i < end; i++) {
            printf("\033c"); // Clear terminal screen
            printLog();
            std::cout << "Processing            : " << fFiles[i].substr(fFiles[i].rfind('A') + 2) << "\n";
            std::cout << "Sampled               : " << counter << " / " << GNFiles;
            std::cout.flush();

            VF vf = readFile(fFiles[i]);
            auto sampledPoints = sampleOne(vf);
            Save(fFiles[i], sampledPoints);
        }
    };

    for (i = 0; i < NChunks; i++)
        threadPool.push_back(std::async(
            std::launch::async,
            sample_iter,
            i * ChunkSize,
            (i + 1) * ChunkSize));

    if (Reminder > 0) {
        threadPool.push_back(std::async(
            std::launch::async,
            sample_iter,
            i * ChunkSize,
            NFiles));
    }

    for (i = 0; i < (int)threadPool.size(); i++)
        threadPool[i].get();
}

void printUsage(void)
{
    printf("Usage : samp_par -p NUMBER_OF_POINTS | -d DATA_PATH | -c CHUNK_SIZE | -r RECURSIVE_CHUNK_SIZE | -h for help");
    exit(2);
}

void printHelp(void)
{
    printf("\033c"); // Clear terminal screen
    printf("\tThis sampler tool samples ASCII-coded mesh files concurrently.\n\n");
    std::cout << "NUMBER_OF_POINTS      :"
              << " number of sample points from a single object (default = 1024)\n";
    std::cout << "DATA_PATH             :"
              << " root folder of dataset (default = /home/yorek/sample/data/ESOGU_ToyDS_A/)\n";
    std::cout << "CHUNK_SIZE            :"
              << " number of files to assign a thread (default = 10)\n";
    std::cout << "RECURSIVE_CHUNK_SIZE  :"
              << " for large files (Faces > 200000) create subthreads for faster sampling(default = 16)\n\n";
    std::cout << "Example Usage : samp_par -p NUMBER_OF_POINTS | -d DATA_PATH | -c CHUNK_SIZE | -r RECURSIVE_CHUNK_SIZE | -h for help\n\n";
}

void MeshSampler::printLog(void)
{
    std::cout << "----------------------> Parameters <----------------------\n";
    std::cout << "NUMBER_OF_POINTS      : " << fNPoints << "\n";
    std::cout << "DATA_PATH             : " << fRoot << "\n";
    std::cout << "CHUNK_SIZE            : " << ChunkSize << "\n";
    std::cout << "RECURSIVE_CHUNK_SIZE  : " << recChunkSize << "\n";
    std::cout << "---------------------->  Progress  <-----------------------\n";
    std::cout.flush();
}
