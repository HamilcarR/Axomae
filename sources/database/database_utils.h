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

   protected:
    int index{};

   protected:
    ImageUpdateMessage() = default;

   public:
    virtual ~ImageUpdateMessage() = default;
    ImageUpdateMessage &operator=(const ImageUpdateMessage &) = default;
    ImageUpdateMessage &operator=(ImageUpdateMessage &&) = default;
    ImageUpdateMessage(ImageUpdateMessage &&) = default;
    ImageUpdateMessage(const ImageUpdateMessage &) = default;
    void setIndex(int id) { index = id; }
    ax_no_discard virtual OPERATION getOperation() const = 0;
    ax_no_discard int getIndex() const { return index; }
  };
  /*image event when an image has been deleted to the DB*/
  class ImageDeleteMessage : public ImageUpdateMessage {
   public:
    ImageDeleteMessage() : ImageUpdateMessage() { index = 0; }
    ax_no_discard OPERATION getOperation() const override { return OPERATION::DELETE; }
  };

  /*******************************************************************************************************************************************/

  /*Image event when an image has been added to the DB*/
  class ImageAddMessage : public ImageUpdateMessage {

   private:
    const QPixmap *value{};
    image::Metadata metadata;

   public:
    ImageAddMessage() : ImageUpdateMessage() { index = 0; }
    ax_no_discard OPERATION getOperation() const override { return OPERATION::ADD; }
    ax_no_discard const QPixmap *getThumbnail() { return value; }
    void setThumbnail(const QPixmap *thumbnail) { value = thumbnail; }
    ax_no_discard image::Metadata getMetadata() { return metadata; }
    void setMetadata(const image::Metadata &metad) { metadata = metad; }
  };

  class ImageSelectedMessage : public ImageUpdateMessage {
   public:
    ImageSelectedMessage() : ImageUpdateMessage() { index = 0; }
    ax_no_discard OPERATION getOperation() const override { return OPERATION::SELECTED; }
  };

}  // namespace database::event

#endif