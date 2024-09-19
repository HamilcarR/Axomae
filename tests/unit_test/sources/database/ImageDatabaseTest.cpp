#include "ImageDatabase.h"
#include "DatabaseBuilderTest.h"
#include <internal/common/math/math_random.h>
#define DATATYPE_LIST uint8_t, float

const int COUNT = 18;
const int TEST_SAMPLES = 20;
const int PIXEL_VALUE_STD = 127;  // Prevents overflow for 1 byte types
template<class TYPE>
class ImageMetadataAndData {
 public:
  explicit ImageMetadataAndData(std::string name) {
    metadata.name = name;
    metadata.width = 2;
    metadata.height = 2;
    metadata.channels = 3;
    metadata.color_corrected = true;
    data.resize(metadata.width * metadata.height * metadata.channels);
    for (auto &elem : data)
      elem = PIXEL_VALUE_STD;
  }

  image::Metadata metadata;
  std::vector<TYPE> data;
};

namespace image_database_test {
  template<class DATATYPE>
  void addImage(IResourceDB<int, image::ThumbnailImageHolder<DATATYPE>> &database) {
    math::random::CPUPseudoRandomGenerator generator;
    bool persistence = generator.randb();
    std::vector<DATATYPE> vec;
    image::Metadata metadata;
    database::image::store(database, persistence);
  }
}  // namespace image_database_test

template<class DATATYPE>
class ImageDatabaseTest final : public DatabaseBuilderTest<int, image::ThumbnailImageHolder<DATATYPE>> {
  using BASETYPE = DatabaseBuilderTest<int, image::ThumbnailImageHolder<DATATYPE>>;

 public:
  explicit ImageDatabaseTest(IResourceDB<int, image::ThumbnailImageHolder<DATATYPE>> &db, int size)
      : DatabaseBuilderTest<int, image::ThumbnailImageHolder<DATATYPE>>(db), total_size(size) {
    buildDatabase();
  }

  void add(image::Metadata &metadata, std::vector<DATATYPE> &vec, bool persistence) {
    database::image::store(BASETYPE::database, persistence);
    total_size++;
  }

 private:
  void buildDatabase() {
    for (int i = 0; i < total_size; i++)
      image_database_test::addImage<DATATYPE>(BASETYPE::database);
  }

  int total_size{};
};

template<class HEAD, class... TAIL>
void test_all_types_add() {
  ImageDatabase<HEAD> database;
  ImageDatabaseTest<HEAD> test(database, TEST_SAMPLES);
  int size = database.size();
  EXPECT_EQ(database.size(), TEST_SAMPLES);
  ImageMetadataAndData<HEAD> template_("");
  test.add(template_.metadata, template_.data, false);
  EXPECT_EQ(database.size(), size + 1);
  ImageMetadataAndData<HEAD> template_name_1("example");
  ImageMetadataAndData<HEAD> template_name_2("example");
  ImageDatabase<HEAD> database2;
  ImageDatabaseTest<HEAD> test2(database2, TEST_SAMPLES);
  size = database2.size();
  test2.add(template_name_1.metadata, template_name_1.data, false);
  size++;
  test2.add(template_name_2.metadata, template_name_2.data, false);
  size++;
  EXPECT_EQ(size, database2.size());
  ImageMetadataAndData<HEAD> template_name_3("example1");
  test2.add(template_name_3.metadata, template_name_3.data, false);
  EXPECT_EQ(++size, database2.size());
  if constexpr (sizeof...(TAIL) > 0)
    test_all_types_add<TAIL...>();
}
TEST(ImageDatabaseTest, add) { test_all_types_add<DATATYPE_LIST>(); }

template<class HEAD, class... TAIL>
void test_all_types_contains() {
  ImageDatabase<HEAD> database;
  ImageDatabaseTest<HEAD> test(database, TEST_SAMPLES);
  int size = database.size();
  for (int i = 0; i < database.size(); i++)
    EXPECT_TRUE(database.contains(i));
  EXPECT_EQ(size, database.size());
  database::Result<int, image::ThumbnailImageHolder<HEAD>> result = database.contains(nullptr);
  EXPECT_EQ(result.object, nullptr);
  for (const auto &elem : database.getConstData())
    EXPECT_EQ(database.contains(elem.second.get()).object, elem.second.get());
  if constexpr (sizeof...(TAIL) > 0)
    test_all_types_contains<TAIL...>();
}
TEST(ImageDatabaseTest, contains) { test_all_types_contains<DATATYPE_LIST>(); }

template<class HEAD, class... TAIL>
void test_all_types_remove() {
  ImageDatabase<HEAD> database;
  ImageDatabaseTest<HEAD> test(database, TEST_SAMPLES);
  int size = database.size();
  for (int i = 0; i < size; i++)
    EXPECT_TRUE(database.remove(i));
  EXPECT_EQ(database.size(), 0);
  ImageDatabase<HEAD> database2;
  ImageDatabaseTest<HEAD> test2(database2, TEST_SAMPLES);
  std::vector<image::ThumbnailImageHolder<HEAD> *> to_delete;
  for (const auto &elem : database2.getConstData())
    to_delete.push_back(elem.second.get());
  for (auto &elem : to_delete)
    EXPECT_TRUE(database2.remove(elem));
  if constexpr (sizeof...(TAIL) > 0)
    test_all_types_remove<TAIL...>();
}
TEST(ImageDatabaseTest, remove) { test_all_types_remove<DATATYPE_LIST>(); }

template<class HEAD, class... TAIL>
void test_all_types_get() {
  ImageDatabase<HEAD> database;
  ImageDatabaseTest<HEAD> test(database, TEST_SAMPLES);
  int size = database.size();
  for (const auto &elem : database.getConstData()) {
    auto *ptr = database.get(elem.first);
    auto it = database.getConstData().find(elem.first);
    EXPECT_EQ(it->second.get(), ptr);
  }
  EXPECT_EQ(size, database.size());
  if constexpr (sizeof...(TAIL) > 0)
    test_all_types_get<TAIL...>();
}

TEST(ImageDatabaseTest, get) { test_all_types_get<DATATYPE_LIST>(); }

template<class HEAD, class... TAIL>
void test_all_types_ffid() {
  ImageDatabase<HEAD> database;
  ImageDatabaseTest<HEAD> test(database, TEST_SAMPLES);
  int ffid = database.firstFreeId();
  EXPECT_EQ(ffid, TEST_SAMPLES);
  database.remove(2);
  ffid = database.firstFreeId();
  EXPECT_EQ(ffid, 2);
  database.remove(0);
  ffid = database.firstFreeId();
  EXPECT_EQ(ffid, 0);

  if constexpr (sizeof...(TAIL) > 0)
    test_all_types_ffid<TAIL...>();
}
TEST(ImageDatabaseTest, firstFreeId) { test_all_types_ffid<DATATYPE_LIST>(); }
