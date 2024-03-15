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
class ImageDatabase : public IntegerResourceDB<image::ThumbnailImageHolder<DATATYPE>>, private IPublisher<database::event::ImageUpdateMessage *> {
  using BaseType = IntegerResourceDB<image::ThumbnailImageHolder<DATATYPE>>;
  using HolderPointer = std::unique_ptr<image::ThumbnailImageHolder<DATATYPE>>;
  using HolderResult = database::Result<int, image::ThumbnailImageHolder<DATATYPE>>;
  using HolderMap = std::map<int, std::unique_ptr<image::ThumbnailImageHolder<DATATYPE>>>;
  using Message = database::event::ImageUpdateMessage *;
  using Subscriber = ISubscriber<Message>;

 private:
  void notifyImageSelected(int index) {
    if (subscribers.empty())
      return;
    database::event::ImageSelectedMessage message;
    message.setIndex(index);
    observer::Data<Message> data{};
    data.data = static_cast<Message>(&message);
    notify(data);
  }

  void notifyImageDelete(int index) {
    if (subscribers.empty())
      return;
    database::event::ImageDeleteMessage message;
    message.setIndex(index);
    observer::Data<Message> data{};
    data.data = static_cast<Message>(&message);
    notify(data);
  }

  void notifyImageAdd(int index) {
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

 public:
  explicit ImageDatabase(controller::ProgressStatus *progress_status = nullptr) { BaseType::progress_manager = progress_status; }

  void purge() override {
    BaseType::purge();
    unique_elements.clear();
  }
  void isSelected(int index) { notifyImageSelected(index); }

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
      AX_ASSERT(result.object != nullptr);
      return result;
    }
    auto elem = BaseType::add(std::move(element), keep);
    notifyImageAdd(elem.id);
    unique_elements.insert(std::pair<std::string, int>(elem.object->metadata().name, elem.id));
    return elem;
  }

  [[nodiscard]] image::Metadata getMetadata(int index) const {
    image::ThumbnailImageHolder<DATATYPE> *holder = BaseType::database_map.at(index).get();
    return holder->metadata();
  }

  [[nodiscard]] const QPixmap &getThumbnail(int index) const {
    image::ThumbnailImageHolder<DATATYPE> *holder = BaseType::database_map.at(index).get();
    return holder->thumbnail();
  }

  void notify(observer::Data<Message> &data) const override {
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
  std::map<std::string, int> unique_elements;  // map all unique images to avoid duplicates
};

namespace database::image {
  template<class TYPE>
  void store(IResourceDB<int, ::image::ThumbnailImageHolder<TYPE>> &database_,
             bool keep,
             std::vector<TYPE> &args,
             const ::image::Metadata &metadata) {
    ASSERT_IS_ARITHMETIC(TYPE);
    std::unique_ptr<::image::ThumbnailImageHolder<TYPE>> raw_image = std::make_unique<::image::ThumbnailImageHolder<TYPE>>(
        args, metadata, database_.getProgressManager());
    database_.add(std::move(raw_image), keep);
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

#endif
