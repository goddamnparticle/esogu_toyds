#include <array>
#include <cassert>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using Point = std::array<double, 3>;
using Face = std::array<int, 3>;
using FileList = std::vector<std::string>;
using pSPoints = std::unique_ptr<double[]>;

extern int counter;
extern int GNFiles;
extern std::vector<std::array<int, 5>> diagReport;

class MeshSampler {
    private:
    FileList fFiles {};
    std::string fRoot {};
    int fNPoints {};
    int ChunkSize;
    int recChunkSize;

    struct VF {
        std::vector<std::array<double, 3>> Verticies {};
        std::vector<std::array<int, 3>> Faces {};
    };

    public:
    MeshSampler(
        int &fNPoints,
        std::string &Root,
        int &ChunkSize,
        int &recChunkSize) : fRoot(Root),
                             fNPoints(fNPoints),
                             ChunkSize(ChunkSize),
                             recChunkSize(recChunkSize) {};
    ~MeshSampler() {};

    public:
    int findFilePaths(void);
    VF readFile(const std::string &path) const;
    pSPoints sampleOne(const VF &vf) const;
    void sampleData(void);
    void Save(const std::string &path, const pSPoints &sampledPoints);
    double TriangleArea(const Point &p1, const Point &p2, const Point &p3) const;
    Point pointInTriangle(const Point &p1, const Point &p2, const Point &p3) const;
    int weightedRandom(std::vector<double> weights) const;
    void printLog(void);
};
