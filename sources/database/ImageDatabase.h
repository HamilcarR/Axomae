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

  void purge() override {
    BaseType::purge();
    unique_elements.clear();
  }

  void clean() override {
    std::vector<typename BaseType::DATABASE::const_iterator> delete_list;
    for (auto it = BaseType::database_map.begin(); it != BaseType::database_map.end(); it++) {
      if (!it->second.isPersistent())
        delete_list.push_back(it);
    }
    Mutex::Lock lock(BaseType::mutex);
    for (auto &elem : delete_list) {
      unique_elements.erase(elem->second.get()->name());
      BaseType::database_map.erase(elem);
    }
  }

  HolderResult add(HolderPointer element, bool keep) override {
    auto it = unique_elements.find(element->metadata().name);
    if (it != unique_elements.end() && !it->first.empty()) {
      HolderResult result;
      result.id = it->second;
      result.object = BaseType::get(result.id);
      assert(result.object != nullptr);
      return result;
    }
    auto elem = BaseType::add(std::move(element), keep);
    notify(observer::Data<Message>());
    unique_elements.insert(std::pair<std::string, int>(elem.object->metadata().name, elem.id));
    return elem;
  }

  [[nodiscard]] int firstFreeId() const override {
    int diff = 0;
    if (BaseType ::database_map.begin()->first > 0)
      return 0;
    for (const auto &elem : BaseType::database_map) {
      if (!elem.second.isValid())
        return elem.first;
      if ((elem.first - diff) > 1)
        return (elem.first + diff) / 2;
      diff = elem.first;
    }
    return BaseType::size();
  }

  [[nodiscard]] image::Metadata getMetadata(int index) const {
    image::RawImageHolder<DATATYPE> *holder = BaseType::database_map.at(index).get();
    return holder->metadata();
  }

  [[nodiscard]] const QPixmap &getThumbnail(int index) const {
    image::RawImageHolder<DATATYPE> *holder = BaseType::database_map.at(index).get();
    return holder->thumbnail();
  }

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
  std::map<std::string, int> unique_elements;  // map all unique images to avoid duplicates
};

namespace database::image {
  template<class TYPE>
  void store(IResourceDB<int, ::image::RawImageHolder<TYPE>> &database_, bool keep, std::vector<TYPE> &args, ::image::Metadata metadata) {
    ASSERT_IS_ARITHMETIC(TYPE);
    auto raw_image = std::make_unique<::image::RawImageHolder<TYPE>>(args, metadata);
    database_.add(std::move(raw_image), keep);
  }

  template<class TYPE>
  void store(IResourceDB<int, ::image::RawImageHolder<TYPE>> &database_, bool keep) {
    ASSERT_IS_ARITHMETIC(TYPE);
    auto raw_image = std::make_unique<::image::RawImageHolder<TYPE>>();
    database_.add(std::move(raw_image), keep);
  }

}  // namespace database::image

using HdrImageDatabase = ImageDatabase<float>;
using RawImageDatabase = ImageDatabase<uint8_t>;

#endif
