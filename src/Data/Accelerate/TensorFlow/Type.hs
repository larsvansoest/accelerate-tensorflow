{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}

module Data.Accelerate.TensorFlow.Type where

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
import qualified Data.Complex
import qualified Data.Word
import qualified Data.ByteString

data TensorTypeDict a where
  TensorTypeDict :: (S.Storable a, TF.TensorDataType S.Vector a, TF.TensorType a) => TensorTypeDict a

data TensorAddDict a where
  TensorAddDict :: (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.ByteString.ByteString, Data.Int.Int16,
                                    Data.Int.Int32, Data.Int.Int64,
                                    Data.Int.Int8, Data.Word.Word16,
                                    Data.Word.Word8, Double, Float] a) => TensorAddDict a

tensorAddDict :: ScalarType a -> TensorAddDict a
tensorAddDict (SingleScalarType single) = case single of { NumSingleType nt -> case nt of
                                                               IntegralNumType it -> case it of
                                                                 TypeInt -> error "no"
                                                                 TypeInt8 -> TensorAddDict
                                                                 TypeInt16 -> TensorAddDict
                                                                 TypeInt32 -> TensorAddDict
                                                                 TypeInt64 -> TensorAddDict
                                                                 TypeWord -> error "no"
                                                                 TypeWord8 -> TensorAddDict
                                                                 TypeWord16 -> TensorAddDict
                                                                 TypeWord32 -> TensorAddDict
                                                                 TypeWord64 -> TensorAddDict
                                                               FloatingNumType ft -> case ft of
                                                                 TypeHalf -> error "no"
                                                                 TypeFloat -> TensorAddDict
                                                                 TypeDouble -> TensorAddDict }
tensorAddDict _ = undefined         

data TensorMulDict a where
  TensorMulDict :: (TF.OneOf '[(Data.Complex.Complex Double),
                                   (Data.Complex.Complex Float), Data.Int.Int16,
                                   Data.Int.Int32, Data.Int.Int64,
                                   Data.Int.Int8, Data.Word.Word16,
                                   Data.Word.Word32, Data.Word.Word64,
                                   Data.Word.Word8, Double, Float] a) => TensorMulDict a

tensorMulDict :: ScalarType a -> TensorMulDict a
tensorMulDict (SingleScalarType single) = case single of { NumSingleType nt -> case nt of
                                                               IntegralNumType it -> case it of
                                                                 TypeInt -> error "no"
                                                                 TypeInt8 -> TensorMulDict
                                                                 TypeInt16 -> TensorMulDict
                                                                 TypeInt32 -> TensorMulDict
                                                                 TypeInt64 -> TensorMulDict
                                                                 TypeWord -> error "no"
                                                                 TypeWord8 -> TensorMulDict
                                                                 TypeWord16 -> TensorMulDict
                                                                 TypeWord32 -> TensorMulDict
                                                                 TypeWord64 -> TensorMulDict
                                                               FloatingNumType ft -> case ft of
                                                                 TypeHalf -> error "no"
                                                                 TypeFloat -> TensorMulDict
                                                                 TypeDouble -> TensorMulDict }
tensorMulDict _ = undefined   

tensorTypeDict :: forall a. ScalarType a -> TensorTypeDict a
tensorTypeDict (SingleScalarType single)                = singleDict single
  where singleDict :: SingleType a -> TensorTypeDict a
        singleDict (NumSingleType (IntegralNumType x)) = case x of
          TypeInt -> error "not a tensortype"
          TypeInt8 -> TensorTypeDict
          TypeInt16 -> TensorTypeDict
          TypeInt32 -> TensorTypeDict
          TypeInt64 -> TensorTypeDict
          TypeWord -> error "not a tensortype"
          TypeWord8 -> TensorTypeDict
          TypeWord16 -> TensorTypeDict
          TypeWord32 -> error "not a tensortype"
          TypeWord64 -> error "not a tensortype"
        singleDict (NumSingleType (FloatingNumType x)) = case x of 
          TypeHalf -> error "not a tensortype"
          TypeFloat -> TensorTypeDict
          TypeDouble -> TensorTypeDict
tensorTypeDict (VectorScalarType (VectorType _ single)) = undefined -- ? singleTensorDict single