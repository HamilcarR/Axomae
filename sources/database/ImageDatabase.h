#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H
#include "Image.h"
#include "RenderingDatabaseInterface.h"
#include "Thumbnail.h"
#include "database_utils.h"
#include "internal/common/Observer.h"
#include "internal/macro/project_macros.h"
#include "internal/memory/MemoryArena.h"
#include <utility>

template<class DATATYPE>
class ImageDatabase : public IntegerResourceDB<image::ThumbnailImageHolder<DATATYPE>>, private IPublisher<database::event::ImageUpdateMessage *> {
  using BaseType = IntegerResourceDB<image::ThumbnailImageHolder<DATATYPE>>;
  using HolderPointer = std::unique_ptr<image::ThumbnailImageHolder<DATATYPE>>;
  using HolderResult = database::Result<int, image::ThumbnailImageHolder<DATATYPE>>;
  using HolderMap = std::map<int, std::unique_ptr<image::ThumbnailImageHolder<DATATYPE>>>;
  using Message = database::event::ImageUpdateMessage *;
  using Subscriber = ISubscriber<Message>;

 private:
  std::map<std::string, int> unique_elements;  // map all unique images to avoid duplicates

 public:
  explicit ImageDatabase(core::memory::MemoryArena<std::byte> *arena = nullptr, controller::ProgressStatus *progress_status = nullptr);
  ~ImageDatabase() override = default;
  void purge() override;
  void isSelected(int index);
  void clean() override;
  HolderResult add(HolderPointer element, bool keep) override;
  ax_no_discard image::Metadata getMetadata(int index) const;
  ax_no_discard const QPixmap &getThumbnail(int index) const;
  void notify(observer::Data<Message> &data) const override;
  void attach(Subscriber &subscriber) override;
  void detach(Subscriber &subscriber) override;

 private:
  void notifyImageSelected(int index);
  void notifyImageDelete(int index);
  void notifyImageAdd(int index);
};

namespace database::image {
  template<class TYPE>
  int store(IResourceDB<int, ::image::ThumbnailImageHolder<TYPE>> &database_, bool keep, std::vector<TYPE> &args, const ::image::Metadata &metadata) {
    ASSERT_IS_ARITHMETIC(TYPE);
    std::unique_ptr<::image::ThumbnailImageHolder<TYPE>> raw_image = std::make_unique<::image::ThumbnailImageHolder<TYPE>>(
        args, metadata, database_.getProgressManager());
    return database_.add(std::move(raw_image), keep).id;
  }

  template<class TYPE>
  void store(IResourceDB<int, ::image::ThumbnailImageHolder<TYPE>> &database_, bool keep) {
    ASSERT_IS_ARITHMETIC(TYPE);
    auto raw_image = std::make_unique<::image::ThumbnailImageHolder<TYPE>>();
    database_.add(std::move(raw_image), keep);
  }
}  // namespace database::image
using HdrImageDatabase = ImageDatabase<float>;
using RawImageDatabase = ImageDatabase<uint8_t>;

/*******************************************************************************************************************************************/

template<class T>
ImageDatabase<T>::ImageDatabase(core::memory::ByteArena *arena, controller::ProgressStatus *progress_status) {
  BaseType::progress_manager = progress_status;
  BaseType::setUpCacheMemory(arena);
}
template<class T>
void ImageDatabase<T>::purge() {
  BaseType::purge();
  unique_elements.clear();
}

template<class T>
void ImageDatabase<T>::isSelected(int index) {
  notifyImageSelected(index);
}

template<class T>
void ImageDatabase<T>::clean() {
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

template<class T>
typename ImageDatabase<T>::HolderResult ImageDatabase<T>::add(HolderPointer element, bool keep) {
  auto it = unique_elements.find(element->metadata.name);
  if (it != unique_elements.end() && !it->first.empty()) {
    HolderResult result;
    result.id = it->second;
    result.object = BaseType::get(result.id);
    AX_ASSERT(result.object != nullptr, "Database query result has invalid object stored");
    return result;
  }
  auto elem = BaseType::add(std::move(element), keep);
  notifyImageAdd(elem.id);
  unique_elements.insert(std::pair<std::string, int>(elem.object->metadata.name, elem.id));
  return elem;
}

template<class T>
image::Metadata ImageDatabase<T>::getMetadata(int index) const {
  image::ThumbnailImageHolder<T> *holder = BaseType::database_map.at(index).get();
  return holder->metadata;
}

template<class T>
const QPixmap &ImageDatabase<T>::getThumbnail(int index) const {
  image::ThumbnailImageHolder<T> *holder = BaseType::database_map.at(index).get();
  return holder->thumbnail();
}

template<class T>
void ImageDatabase<T>::notify(observer::Data<Message> &data) const {
  for (Subscriber *A : subscribers)
    A->notified(data);
}

template<class T>
void ImageDatabase<T>::attach(Subscriber &subscriber) {
  subscribers.push_back(&subscriber);
}

template<class T>
void ImageDatabase<T>::detach(Subscriber &subscriber) {
  for (auto it = subscribers.begin(); it != subscribers.end(); it++)
    if (*it == &subscriber)
      subscribers.erase(it);
}

template<class T>
void ImageDatabase<T>::notifyImageSelected(int index) {
  if (subscribers.empty())
    return;
  database::event::ImageSelectedMessage message;
  message.setIndex(index);
  observer::Data<Message> data{};
  data.data = static_cast<Message>(&message);
  notify(data);
}

template<class T>
void ImageDatabase<T>::notifyImageDelete(int index) {
  if (subscribers.empty())
    return;
  database::event::ImageDeleteMessage message;
  message.setIndex(index);
  observer::Data<Message> data{};
  data.data = static_cast<Message>(&message);
  notify(data);
}

template<class T>
void ImageDatabase<T>::notifyImageAdd(int index) {
  if (subscribers.empty())
    return;
  database::event::ImageAddMessage message;
  message.setMetadata(getMetadata(index));
  message.setIndex(index);
  message.setThumbnail(&getThumbnail(index));
  observer::Data<Message> data{};
  data.data = static_cast<Message>(&message);
  notify(data);
}

#endif
