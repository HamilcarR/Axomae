#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H
#include "Axomae_macros.h"
#include "Factory.h"
#include "Image.h"
#include "Observer.h"
#include "RenderingDatabaseInterface.h"
#include "Thumbnail.h"
#include "database_utils.h"
#include <utility>

template<class DATATYPE>
class ImageDatabase : public IResourceDB<int, image::RawImageHolder<DATATYPE>>, public IPublisher<database::event::IconUpdateMessage> {
  using BaseType = IResourceDB<int, image::RawImageHolder<DATATYPE>>;
  using HolderPointer = std::unique_ptr<image::RawImageHolder<DATATYPE>>;
  using HolderResult = database::Result<int, image::RawImageHolder<DATATYPE>>;
  using HolderMap = std::map<int, std::unique_ptr<image::RawImageHolder<DATATYPE>>>;
  using Message = database::event::IconUpdateMessage;
  using Subscriber = ISubscriber<Message>;

 private:
  void notifyIconUpdate(int index) {
    Message message;
    message.metadata = getMetadata(index);
    message.index = index;
    message.value = getThumbnail(index);
    observer::Data<Message> data;
    data.data = message;
    notify(data);
  }

 public:
  ImageDatabase() = default;

  void clean() override {
    Mutex::Lock lock(BaseType::mutex);
    std::vector<int> deletion;
    for (const auto &A : BaseType::database_map) {
      if (A.first >= 0)
        deletion.push_back(A.first);
    }
    for (int A : deletion)
      BaseType::database_map.erase(A);
  }

  HolderResult add(HolderPointer element, bool keep) override {
    Mutex::Lock lock(BaseType::mutex);
    if (keep) {
      int index = -1;
      while (BaseType::database_map[index] != nullptr)
        index--;
      BaseType::database_map[index] = std::move(element);
      return {index, BaseType::database_map[index].get()};
    } else {
      int index = 0;
      while (BaseType::database_map[index] != nullptr)
        index++;
      BaseType::database_map[index] = std::move(element);
      notify(observer::Data<Message>());
      return {index, BaseType::database_map[index].get()};
    }
  }

  [[nodiscard]] image::Metadata getMetadata(int index) const {
    image::RawImageHolder<DATATYPE> *holder = BaseType::database_map.at(index).get();
    return holder->metadata();
  }

  [[nodiscard]] QPixmap getThumbnail(int index) const {
    image::RawImageHolder<DATATYPE> *holder = BaseType::database_map.at(index).get();
    return holder->thumbnail();
  }

  const HolderMap &getConstData() const override { return BaseType::database_map; }

  void notify(observer::Data<Message> data) const override {
    for (Subscriber *A : subscribers)
      A->notified(data);
  }

  void attach(Subscriber &subscriber) override { subscribers.push_back(&subscriber); }

  void detach(Subscriber &subscriber) override {
    for (auto it = subscribers.begin(); it != subscribers.end(); it++)
      if (*it == &subscriber)
        subscribers.erase(it);
  }

 private:
  std::vector<Subscriber *> subscribers;
};

namespace database {
  template<class TYPE>
  void store(IResourceDB<int, image::RawImageHolder<TYPE>> &database_, bool keep, std::vector<TYPE> &args, image::Metadata metadata) {
    ASSERT_IS_ARITHMETIC(TYPE);
    auto raw_image = std::make_unique<image::RawImageHolder<TYPE>>(args, metadata);
    database_.add(std::move(raw_image), keep);
  }
}  // namespace database

using HdrImageDatabase = ImageDatabase<float>;

#endif
