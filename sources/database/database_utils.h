#ifndef DATABASE_UTILS_H
#define DATABASE_UTILS_H

#include "Image.h"
#include "constants.h"
#include <QModelIndex>
#include <QVariant>

namespace database::event {
  class ImageUpdateMessage {
   public:
    enum OPERATION : unsigned { DELETE = 0, ADD = 1, SELECTED = 2 };
    virtual ~ImageUpdateMessage() = default;
    ImageUpdateMessage &operator=(const ImageUpdateMessage &) = default;
    ImageUpdateMessage &operator=(ImageUpdateMessage &&) = default;
    ImageUpdateMessage(ImageUpdateMessage &&) = default;
    ImageUpdateMessage(const ImageUpdateMessage &) = default;
    void setIndex(int id) { index = id; }
    [[nodiscard]] virtual OPERATION getOperation() const = 0;
    [[nodiscard]] int getIndex() const { return index; }

   protected:
    ImageUpdateMessage() = default;

   protected:
    int index{};
  };
  /*image event when an image has been deleted to the DB*/
  class ImageDeleteMessage : public ImageUpdateMessage {
   public:
    ImageDeleteMessage() : ImageUpdateMessage() { index = 0; }
    [[nodiscard]] OPERATION getOperation() const override { return OPERATION::DELETE; }
  };

  /*Image event when an image has been added to the DB*/
  class ImageAddMessage : public ImageUpdateMessage {
   public:
    ImageAddMessage() : ImageUpdateMessage() { index = 0; }
    [[nodiscard]] OPERATION getOperation() const override { return OPERATION::ADD; }
    [[nodiscard]] const QPixmap *getThumbnail() { return value; }
    void setThumbnail(const QPixmap *thumbnail) { value = thumbnail; }
    [[nodiscard]] image::Metadata getMetadata() { return metadata; }
    void setMetadata(image::Metadata metad) { metadata = metad; }

   private:
    const QPixmap *value{};
    image::Metadata metadata;
  };

  class ImageSelectedMessage : public ImageUpdateMessage {
   public:
    ImageSelectedMessage() : ImageUpdateMessage() { index = 0; }
    [[nodiscard]] OPERATION getOperation() const override { return OPERATION::SELECTED; }
  };

}  // namespace database::event

#endif