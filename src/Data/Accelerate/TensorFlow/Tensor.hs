{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Accelerate.TensorFlow.Tensor where

import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.Array.Unique (UniqueArray(UniqueArray), withUniqueArrayPtr)
import Data.Array.Accelerate.Lifetime
import GHC.ForeignPtr
import Control.Monad.IO.Class (liftIO)
import Data.Array.Accelerate.Type (ScalarType (..), SingleType (NumSingleType), NumType (IntegralNumType, FloatingNumType), IntegralType (..), FloatingType (..), VectorType (..))

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import qualified TensorFlow.Internal.FFI as FFI
import qualified Data.Vector as V
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Foreign (Ptr, castPtr, Word8)
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(Refl), matchScalarType )
import qualified Data.Functor.Identity as TF
import qualified Data.Vector.Storable as S
import Text.Printf

data TensorTypeDict a where
  TensorTypeDict :: TF.TensorType a => TensorTypeDict a

data TensorDataTypeDict a where
  TensorDataTypeDict :: (TF.TensorDataType S.Vector a) => TensorDataTypeDict a

data StorableDict a where
  StorableDict :: (S.Storable a) => StorableDict a

storableDict :: ScalarType a -> StorableDict a
storableDict (SingleScalarType single)                = storableDict' single
storableDict (VectorScalarType (VectorType _ single)) = undefined -- ? singleTensorDict single

storableDict' :: SingleType a -> StorableDict a
storableDict' (NumSingleType (IntegralNumType x)) = case x of
  TypeInt -> StorableDict
  TypeInt8 -> StorableDict
  TypeInt16 -> StorableDict
  TypeInt32 -> StorableDict
  TypeInt64 -> StorableDict
  TypeWord -> StorableDict
  TypeWord8 -> StorableDict
  TypeWord16 -> StorableDict
  TypeWord32 -> StorableDict
  TypeWord64 ->StorableDict
storableDict' (NumSingleType (FloatingNumType x)) = case x of
  TypeHalf -> StorableDict
  TypeFloat -> StorableDict
  TypeDouble -> StorableDict

tensorDataTypeDict :: ScalarType a -> TensorDataTypeDict a
tensorDataTypeDict (SingleScalarType single)                = tensorDataTypeDict' single
tensorDataTypeDict (VectorScalarType (VectorType _ single)) = undefined -- ? singleTensorDict single

tensorDataTypeDict' :: SingleType a -> TensorDataTypeDict a
tensorDataTypeDict' (NumSingleType (IntegralNumType x)) = case x of
  TypeInt -> error "not a tensortype"
  TypeInt8 -> TensorDataTypeDict
  TypeInt16 -> TensorDataTypeDict
  TypeInt32 -> TensorDataTypeDict
  TypeInt64 -> TensorDataTypeDict
  TypeWord -> error "not a tensortype"
  TypeWord8 -> TensorDataTypeDict
  TypeWord16 -> TensorDataTypeDict
  TypeWord32 -> error "not a tensortype"
  TypeWord64 -> error "not a tensortype"
tensorDataTypeDict' (NumSingleType (FloatingNumType x)) = case x of
  TypeHalf -> error "not a tensortype"
  TypeFloat -> TensorDataTypeDict
  TypeDouble -> TensorDataTypeDict

scalarTensorDict :: ScalarType a -> TensorTypeDict a
scalarTensorDict (SingleScalarType single)                = singleTensorDict single
scalarTensorDict (VectorScalarType (VectorType _ single)) = undefined -- ? singleTensorDict single

singleTensorDict :: SingleType a -> TensorTypeDict a
singleTensorDict (NumSingleType (IntegralNumType x)) = case x of
  TypeInt -> error "not a tensortype"
  TypeInt8 -> TensorTypeDict
  TypeInt16 -> TensorTypeDict
  TypeInt32 -> TensorTypeDict
  TypeInt64 -> TensorTypeDict
  TypeWord -> error "not a tensortype"
  TypeWord8 -> TensorTypeDict
  TypeWord16 -> TensorTypeDict
  TypeWord32 -> TensorTypeDict
  TypeWord64 -> TensorTypeDict
singleTensorDict (NumSingleType (FloatingNumType x)) = case x of 
  TypeHalf -> error "not a tensortype"
  TypeFloat -> TensorTypeDict
  TypeDouble -> TensorTypeDict

-- 2

toTFShape :: ShapeR sh -> sh -> TF.Shape
toTFShape shR sh = TF.Shape $ fromIntegral <$> shapeToList shR sh

-- build ipv value
-- hoe maak ik de buffer ervan?
-- hoe case match ik op TensorType?
-- fromBuffer :: ShapeR sh -> ScalarType t -> sh -> Buffer t -> TF.Tensor TF.Build t
-- fromBuffer shR t sh buffer 
--   | TensorTypeDict <- scalarTensorDict t
--   = TF.constant (toTFShape shR sh) $ bufferToList t (size shR sh) buffer

fromBuffer :: ShapeR sh -> 
  ScalarType t -> 
  sh -> 
  Buffer t ->
  TF.Session (S.Vector t) -- V instead of S?
fromBuffer shR t sh buffer 
  | TensorTypeDict <- scalarTensorDict t 
  , TensorDataTypeDict <- tensorDataTypeDict t
  , StorableDict <- storableDict t
  = TF.runSession $ do
    let shape = toTFShape shR sh
    x <- TF.placeholder shape
    let vec = S.fromList (bufferToList t (size shR sh) buffer)
    let feeds = [ TF.feed x $ TF.encodeTensorData shape vec ]
    TF.runWithFeeds feeds x

toBuffer :: ScalarType t -> IO (V.Vector t) -> IO (Buffer t)
toBuffer t v = undefined

-- 1 Data.Array.Accelerate.AST.Execute
-- executeAfunSchedule :: GFunctionR t -> sched kernel () (Scheduled sched t) -> IOFun (Scheduled sched t)
